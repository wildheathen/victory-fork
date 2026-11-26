//! Approach after XCL 3ecf4cf: preserve both halves of a guest wave and make
//! its scalar decisions once. Here a host lane keeps two sets of SSA values.
//! Lane 7, for example, executes guest lanes 7 and 39. A lane read can reach
//! either set without a barrier inside a branch.

use super::*;
use crate::emit::shader::{Shader, flags};

type Pair = [V; 2];
type Result<T> = std::result::Result<T, &'static str>;

/// A vector instruction can generate its own control flow, for example a
/// guarded store. Each half follows its own edges to the instruction's
/// join before execution continues with the next guest instruction.
#[derive(Clone, Copy)]
struct Region {
    head: usize,
    join: usize,
    middle: usize,
}

struct Blocks {
    ids: Vec<[usize; 2]>,
    starts: Vec<Option<Region>>,
    ends: Vec<Option<Region>>,
    inside: Vec<Option<Region>>,
}

impl Blocks {
    fn new(m: &mut Module, source: &[Block], regions: &[(usize, usize)]) -> Result<Self> {
        let n = source.len();
        let mut dest = Self {
            ids: (0..n).map(|_| [m.new_block(); 2]).collect(),
            starts: vec![None; n],
            ends: vec![None; n],
            inside: vec![None; n],
        };
        for &(head, join) in regions {
            let region = Region {
                head,
                join,
                middle: m.new_block(),
            };
            if dest.starts[head].replace(region).is_some()
                || dest.ends[join].replace(region).is_some()
            {
                return Err("overlapping lane instructions");
            }
            let mut pending = successors(source[head].term).to_vec();
            while let Some(b) = pending.pop() {
                if b == join || dest.inside[b].is_some_and(|r| r.head == head) {
                    continue;
                }
                if b == head
                    || dest.inside[b].replace(region).is_some()
                    || source[b].term.is_none_or(|t| matches!(t, Term::Ret))
                {
                    return Err("a lane instruction without a single join");
                }
                dest.ids[b][1] = m.new_block();
                pending.extend_from_slice(&successors(source[b].term));
            }
        }
        Ok(dest)
    }

    fn from(&self, b: usize, bank: usize) -> usize {
        if bank == 1
            && let Some(r) = self.starts[b]
        {
            return r.middle;
        }
        self.ids[b][bank]
    }

    fn to(&self, from: usize, to: usize, bank: usize) -> usize {
        if bank == 0
            && let Some(r) = self.starts[from].or(self.inside[from])
            && to == r.join
        {
            return r.middle;
        }
        self.ids[to][bank]
    }
}

struct Values(Vec<Option<Pair>>);

impl Values {
    fn get(&self, v: V) -> Result<Pair> {
        match v {
            V::Inst(i) => self.0[i as usize].ok_or("a value outside its wave instruction"),
            _ => Ok([v; 2]),
        }
    }

    fn operation(&self, op: &Op, bank: usize) -> Result<Op> {
        let mut copy = op.clone();
        let mut valid = Ok(());
        opt::map_operands(&mut copy, |v| match self.get(v) {
            Ok(pair) => pair[bank],
            Err(e) => {
                valid = Err(e);
                v
            }
        });
        valid.map(|()| copy)
    }
}

/// Visit definitions before their uses. Loop-carried phis are wired after
/// both copies of their backedges have been emitted.
fn order(blocks: &[Block]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut seen = vec![false; blocks.len()];
    let mut pending = vec![(0, false)];
    while let Some((b, leaving)) = pending.pop() {
        if leaving {
            out.push(b);
        } else if !std::mem::replace(&mut seen[b], true) {
            pending.push((b, true));
            pending.extend(successors(blocks[b].term).iter().map(|&s| (s, false)));
        }
    }
    out.reverse();
    out
}

/// Number complete physical waves once at entry. The API does not promise
/// that a physical wave owns consecutive flattened thread IDs. The first
/// guest LDS word can serve as the counter until guest instructions begin.
fn thread_numbers(s: &mut Shader, group: u32) -> (Pair, Pair) {
    let b = s.entry;
    let lane = s.op(b, "waveGetLaneIndex", Attr::ReadOnly, s.t.i32, 111, &[]);
    let k32 = s.m.i32(32);
    let high = s.m.bin(b, Bin::Add, lane, k32);
    let lanes = [lane, high];
    if group == 64 {
        return (lanes, lanes);
    }
    s.lds_dwords = s.lds_dwords.max(1);
    let lds = s.lds();
    let zero = s.m.i32(0);
    let flat = s.op(
        b,
        "flattenedThreadIdInGroup.i32",
        Attr::ReadNone,
        s.t.i32,
        96,
        &[],
    );
    let first = s.m.cmp(b, pred::EQ, flat, zero);
    let init = s.m.new_block();
    let count = s.m.new_block();
    s.m.terminate(b, Term::CondBr(first, init, count));
    let p = s.m.element(init, lds, zero);
    s.m.store(init, p, zero);
    s.m.terminate(init, Term::Br(count));
    let mode = s.m.i32(9);
    s.op(count, "barrier", Attr::NoDuplicate, s.t.void, 80, &[mode]);
    let leader = s.m.cmp(count, pred::EQ, lane, zero);
    let stride = s.m.i32(64);
    let add = s.m.select(count, leader, stride, zero);
    let p = s.m.element(count, lds, zero);
    let ticket = s.m.atomic(count, Rmw::Add, p, add);
    let base = s.op(
        count,
        "waveReadLaneFirst.i32",
        Attr::NoUnwind,
        s.t.i32,
        118,
        &[ticket],
    );
    s.op(count, "barrier", Attr::NoDuplicate, s.t.void, 80, &[mode]);
    let flat = lanes.map(|l| s.m.bin(count, Bin::Add, base, l));
    s.entry = count;
    (lanes, flat)
}

/// Numbering several physical waves uses one group rendezvous before any guest
/// instruction. A diagnostic run that disables group waits can still pair a
/// single wave without that rendezvous.
fn guest_threads(shape: [u32; 3], group_waits: bool) -> Result<u32> {
    let group = shape
        .into_iter()
        .try_fold(1u32, u32::checked_mul)
        .ok_or("invalid compute group")?;
    if group == 0 || group > 1024 || !group.is_multiple_of(64) {
        return Err("paired execution needs complete guest waves");
    }
    if group > 64 && !group_waits {
        return Err("numbering several paired waves requires group synchronization");
    }
    Ok(group)
}
/// Convert one complete compute group. Partial guest waves need an entry
/// execution mask for the missing lanes. Leave those to the other lowering.
pub(crate) fn run(
    s: &mut Shader,
    regions: &[(usize, usize)],
    wave_exits: &[Option<usize>],
) -> Result<()> {
    let shape = s.threads.ok_or("missing compute group")?;
    let group = guest_threads(shape, crate::emit::group_sync() != 0)?;
    let old = std::mem::take(&mut s.m.blocks);
    let instructions = std::mem::take(&mut s.m.insts);
    s.m.tags.clear();
    let mut values = Values(vec![None; instructions.len()]);
    s.entry = s.m.new_block();
    let (lanes, flat) = thread_numbers(s, group);
    let dest = Blocks::new(&mut s.m, &old, regions)?;
    s.m.terminate(s.entry, Term::Br(dest.ids[0][0]));
    s.entry = 0;
    let mut whole_wave = vec![false; old.len()];
    for &b in wave_exits.iter().flatten() {
        whole_wave[b] = true;
    }
    // A branch on a value both halves compute alike is the wave's too: the
    // dispatch spread's exit of the groups past the asked count
    // (`Lower::cs_spread_prologue`) branches on the group id and constants
    // in the entry block, and was refused as a lane branch.
    let same = uniform(&s.m, &instructions);
    for (b, block) in old.iter().enumerate() {
        if let Some(Term::CondBr(c, ..)) = block.term
            && is_uniform(&same, c)
        {
            whole_wave[b] = true;
        }
    }
    let mut pending = Vec::new();
    for b in order(&old) {
        let block = &old[b];
        // Guest scalar branches take one path for the wave. A branch
        // introduced by a single vector instruction can differ by lane.
        if matches!(block.term, Some(Term::CondBr(..)))
            && !whole_wave[b]
            && dest.starts[b].is_none()
            && dest.inside[b].is_none()
        {
            return Err("a conditional outside the guest wave and lane instruction regions");
        }
        let ids = dest.ids[b];
        for &i in &block.phis {
            let ty = instructions[i as usize].ty;
            let low = dest.ends[b].map_or(ids[0], |r| r.middle);
            let pair = [s.m.phi(low, ty), s.m.phi(ids[1], ty)];
            values.0[i as usize] = Some(pair);
            pending.push((i, pair));
        }
        for &i in &block.body {
            let Inst { ty, op } = instructions[i as usize].clone();
            let paired = match &op {
                Op::Call(f, args) => {
                    let opcode = args
                        .first()
                        .and_then(|&a| s.m.int_value(a))
                        .ok_or("unknown DXIL call")?;
                    match opcode {
                        111 => Some(lanes),
                        112 => Some([s.m.i32(64); 2]),
                        96 => Some(flat),
                        95 => {
                            let dim = s.m.int_value(args[1]).ok_or("dynamic thread dimension")?;
                            let dim = usize::try_from(dim)
                                .ok()
                                .filter(|&d| d < 3)
                                .ok_or("invalid thread dimension")?;
                            let stride = s.m.i32(shape[..dim].iter().product());
                            let size = s.m.i32(shape[dim]);
                            Some(std::array::from_fn(|k| {
                                let at = s.m.bin(ids[k], Bin::UDiv, flat[k], stride);
                                s.m.bin(ids[k], Bin::URem, at, size)
                            }))
                        }
                        80 if ids[0] == ids[1] => {
                            Some([s.m.call(ids[0], V::Func(*f), args.clone()); 2])
                        }
                        80 => return Err("a group barrier inside a guarded lane instruction"),
                        110 | 113..=123 if ids[0] == ids[1] => {
                            Some(wave(s, ids[0], *f, opcode, args, &values)?)
                        }
                        110 | 113..=123 => {
                            return Err("a wave operation inside a guarded lane instruction");
                        }
                        _ => None,
                    }
                }
                Op::Extract(V::Inst(a), field)
                    if matches!(&instructions[*a as usize].op,
                    Op::Call(_, args) if s.m.int_value(args[0]) == Some(116)) =>
                {
                    let ballot = values.get(V::Inst(*a))?;
                    let word = if *field < 2 {
                        s.m.extract(ids[0], ballot[*field as usize], 0)
                    } else {
                        s.m.i32(0)
                    };
                    // A guarded consumer in the upper copy also needs a dominating definition.
                    Some([
                        word,
                        if ids[0] == ids[1] || *field >= 2 {
                            word
                        } else {
                            s.m.extract(ids[1], ballot[*field as usize], 0)
                        },
                    ])
                }
                _ => None,
            };
            let pair = match paired {
                Some(pair) => pair,
                None => [
                    s.m.push(ids[0], ty, values.operation(&op, 0)?),
                    s.m.push(ids[1], ty, values.operation(&op, 1)?),
                ],
            };
            values.0[i as usize] = Some(pair);
        }
        let banks = if dest.starts[b].is_some() || dest.inside[b].is_some() {
            2
        } else {
            1
        };
        for k in 0..banks {
            let target = |next| dest.to(b, next, k);
            let term = match block.term.ok_or("unterminated compute block")? {
                Term::Br(t) => Term::Br(target(t)),
                Term::CondBr(c, t, f) => Term::CondBr(values.get(c)?[k], target(t), target(f)),
                Term::Ret => Term::Ret,
            };
            s.m.terminate(dest.from(b, k), term);
        }
    }
    for (i, pair) in pending {
        let Op::Phi(edges) = &instructions[i as usize].op else {
            unreachable!()
        };
        for (k, phi) in pair.into_iter().enumerate() {
            let edges = edges
                .iter()
                .map(|&(v, pred)| Ok((values.get(v)?[k], dest.from(pred, k))))
                .collect::<Result<Vec<_>>>()?;
            s.m.set_phi(phi, edges);
        }
    }
    for (_, handle) in s
        .root_cbvs
        .iter_mut()
        .chain(&mut s.root_srvs)
        .chain(&mut s.root_uavs)
    {
        *handle = values.get(*handle)?[0];
    }
    // Replaced system values can leave an unused DXIL declaration even when
    // the optional optimization passes are disabled.
    s.m.drop_unused_functions();
    s.flags |= flags::WAVE_OPS;
    s.wave_size = Some(32);
    s.threads = Some([group / 2, 1, 1]);
    Ok(())
}
/// Instructions whose value is the same for every lane of the group: built
/// only from constants, group ids, root values and loads at such addresses,
/// never from a lane's identity, a wave operation, an atomic, an LDS read or
/// a phi. Operands come before their uses except through phis, so one pass
/// in order settles it (an operand defined later counts as not uniform).
fn uniform(m: &Module, insts: &[Inst]) -> Vec<bool> {
    let mut same = vec![false; insts.len()];
    for (i, inst) in insts.iter().enumerate() {
        let all = |vs: &[V]| vs.iter().all(|&v| is_uniform(&same, v));
        same[i] = match &inst.op {
            Op::Bin(_, a, b) | Op::Cmp(_, a, b) => all(&[*a, *b]),
            Op::Cast(_, a) | Op::Extract(a, _) => all(&[*a]),
            Op::Select(c, a, b) => all(&[*c, *a, *b]),
            Op::Gep(_, base, idx) => is_uniform(&same, *base) && all(idx),
            Op::Call(_, args) => match args.first().and_then(|&a| m.int_value(a)) {
                // Thread and lane ids, wave operations, UAV atomics.
                Some(93 | 95 | 96 | 110 | 111 | 113..=123 | 78 | 79) | None => false,
                Some(_) => all(args),
            },
            Op::Phi(_) | Op::Load(_) | Op::Store(..) | Op::AtomicRmw(..) | Op::CmpXchg(..) => false,
        };
    }
    same
}

fn is_uniform(same: &[bool], v: V) -> bool {
    match v {
        V::Inst(i) => same.get(i as usize).copied().unwrap_or(false),
        _ => true,
    }
}

/// A guest instruction sees both register sets while all physical lanes are
/// present. In particular a mask whose upper half is empty contributes zero
/// to the upper word of the ballot.
fn wave(s: &mut Shader, b: usize, f: u32, opcode: i64, args: &[V], vals: &Values) -> Result<Pair> {
    let arg = |i: usize| {
        args.get(i)
            .ok_or("missing wave operand")
            .and_then(|&v| vals.get(v))
    };
    let call = |s: &mut Shader, k: usize| -> Result<V> {
        let operands = args
            .iter()
            .map(|&v| Ok(vals.get(v)?[k]))
            .collect::<Result<Vec<_>>>()?;
        Ok(s.m.call(b, V::Func(f), operands))
    };
    Ok(match opcode {
        110 => [call(s, 0)?, s.m.bool(false)],
        113 | 114 => {
            let halves = [call(s, 0)?, call(s, 1)?];
            [s.m.bin(
                b,
                if opcode == 113 { Bin::Or } else { Bin::And },
                halves[0],
                halves[1],
            ); 2]
        }
        116 | 122 | 123 => [call(s, 0)?, call(s, 1)?],
        117 => {
            let src = arg(1)?;
            let indices = arg(2)?;
            let mask = s.m.i32(31);
            let mut read = |idx: V| {
                let lane = s.m.bin(b, Bin::And, idx, mask);
                let lower = s.m.cmp(b, pred::IULE, idx, mask);
                let halves = src.map(|v| s.m.call(b, V::Func(f), vec![args[0], v, lane]));
                s.m.select(b, lower, halves[0], halves[1])
            };
            [read(indices[0]), read(indices[1])]
        }
        118 => [call(s, 0)?; 2],
        119..=121 => {
            let halves = [call(s, 0)?, call(s, 1)?];
            let kind = s.m.int_value(args[2]).ok_or("dynamic wave operation")?;
            let ty = s.m.type_of(halves[0]);
            let left = if opcode == 121 {
                let name = s.m.funcs[f as usize]
                    .name
                    .replace("dx.op.wavePrefixOp.", "waveActiveOp.");
                s.op(
                    b,
                    &name,
                    Attr::NoUnwind,
                    ty,
                    119,
                    &[arg(1)?[0], args[2], args[3]],
                )
            } else {
                halves[0]
            };
            let right = halves[1];
            let joined = match (opcode, kind) {
                (120, 0) => s.m.bin(b, Bin::And, left, right),
                (120, 1) => s.m.bin(b, Bin::Or, left, right),
                (120, 2) => s.m.bin(b, Bin::Xor, left, right),
                (120, _) => return Err("unknown wave bit operation"),
                (_, 0) => s.m.bin(b, Bin::Add, left, right),
                (_, 1) => s.m.bin(b, Bin::Mul, left, right),
                (_, 2 | 3) => {
                    let (name, min, max) = match s.m.types[ty.0 as usize] {
                        TypeDef::Float => ("binary.f32", 36, 35),
                        TypeDef::Double => ("binary.f64", 36, 35),
                        TypeDef::Int(32) if s.m.int_value(args[3]) == Some(1) => {
                            ("binary.i32", 40, 39)
                        }
                        TypeDef::Int(32) => ("binary.i32", 38, 37),
                        _ => return Err("unknown wave arithmetic type"),
                    };
                    s.op(
                        b,
                        name,
                        Attr::ReadNone,
                        ty,
                        if kind == 2 { min } else { max },
                        &[left, right],
                    )
                }
                _ => return Err("unknown wave arithmetic operation"),
            };
            if opcode == 121 {
                [halves[0], joined]
            } else {
                [joined; 2]
            }
        }
        _ => return Err("unsupported paired wave instruction"),
    })
}

#[cfg(test)]
#[test]
fn paired_group_admission_respects_lane_count_and_disabled_waits() {
    assert_eq!(guest_threads([8, 8, 1], false), Ok(64));
    assert_eq!(guest_threads([8, 8, 2], true), Ok(128));
    assert!(guest_threads([8, 8, 2], false).is_err());
    for shape in [
        [48, 1, 1],
        [96, 1, 1],
        [0, 1, 1],
        [1025, 1, 1],
        [u32::MAX, 2, 1],
    ] {
        assert!(guest_threads(shape, true).is_err());
    }
}

#[cfg(test)]
#[test]
fn a_branch_on_the_group_id_is_uniform_and_one_on_the_lane_is_not() {
    let mut s = Shader::new(false);
    let b = s.entry;
    let zero = s.m.i32(0);
    let asked = s.m.i32(7);
    // As the dispatch spread's exit: the group id against a count.
    let group = s.op(b, "groupId.i32", Attr::ReadNone, s.t.i32, 94, &[zero]);
    let past = s.m.cmp(b, pred::IULT, group, asked);
    let lane = s.op(b, "waveGetLaneIndex", Attr::ReadOnly, s.t.i32, 111, &[]);
    let mixed = s.m.bin(b, Bin::Add, group, lane);
    let low = s.m.cmp(b, pred::IULT, mixed, asked);
    // A phi is never taken as uniform, whatever it joins.
    let next = s.m.new_block();
    s.m.terminate(b, Term::Br(next));
    let phi = s.m.phi(next, s.t.i32);
    s.m.set_phi(phi, vec![(group, b)]);
    let same = uniform(&s.m, &s.m.insts);
    assert!(is_uniform(&same, past));
    assert!(is_uniform(&same, zero));
    assert!(!is_uniform(&same, lane));
    assert!(!is_uniform(&same, low));
    assert!(!is_uniform(&same, phi));
}
