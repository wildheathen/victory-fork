//! Synthetic compute programs for the two guest lanes carried by a physical lane.

use super::*;
use windows::Win32::Graphics::Direct3D12::{
    D3D12_FEATURE_D3D12_OPTIONS1, D3D12_FEATURE_DATA_D3D12_OPTIONS1,
};

fn device() -> Option<Host> {
    let host = match Host::new(false) {
        Ok(host) => host,
        Err(e) => {
            eprintln!("paired GPU readback skipped: {e}");
            return None;
        }
    };
    let mut waves = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the feature, structure and size agree.
    unsafe {
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut waves).cast(),
                size_of_val(&waves) as u32,
            )
            .expect("device wave range")
    };
    if !waves.WaveOps.as_bool() || !(waves.WaveLaneCountMin..=waves.WaveLaneCountMax).contains(&32)
    {
        eprintln!(
            "paired GPU readback skipped: wave range {}..{}",
            waves.WaveLaneCountMin, waves.WaveLaneCountMax
        );
        return None;
    }
    Some(host)
}

fn paired_fixture(native: &NativeProgram) -> HostShader {
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let bytes = super::super::container::build(&[]);
    let c = crate::Container::parse(&bytes).unwrap();
    let (shader, wave) =
        super::super::emit_paired(&c, native, &p, &st).expect("paired translation");
    assert_eq!(wave, Some(32));
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("paired DXIL validation");
    }
    shader
}

fn readback(host: &Host, shader: &HostShader, count: u32, words: u32) -> Vec<u32> {
    let (root, pso) = host
        .compute_pipeline(shader)
        .expect("paired compute pipeline");
    let mut data = vec![0; shader.user_data_dwords() as usize];
    data[0] = count;
    host.dispatch(
        &root,
        &pso,
        &Dispatch {
            roots: vec![(0, Root::Constants(data))],
            views: &[View::RawUav(words)],
            samplers: &[],
            groups: [1, 1, 1],
        },
    )
    .expect("paired dispatch")
    .remove(0)
}

#[test]
fn paired_wave_reduction_counts_selected_lanes_in_both_halves() {
    let shader = paired_fixture(&program(0xBF88_0000));
    let _one = super::super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some(host) = device() else { return };
    for count in [0u32, 1, 31, 32, 33, 63, 64] {
        let expected = (0..count).fold(0u32, |bits, i| bits | (1 << (i / 2)));
        assert_eq!(
            readback(&host, &shader, count, 1),
            [expected],
            "{count} live lanes"
        );
    }
    eprintln!("paired wave reduction passed GPU readback");
}

/// Each wave selects lane 43 as a scalar. Selected lanes add that scalar
/// to an LDS word written by a different lane. With two guest waves per
/// group the word comes from the other wave after a guest group barrier.
fn shared(threads: u32) -> NativeProgram {
    let body = [
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0xD834_0000,
        0x0000_0001,                                         // ds_write_b32 v1, v0
        0xBF8A_0000,                                         // s_barrier
        0x3A04_0000 | if threads == 64 { 160 } else { 192 }, // v_xor_b32 v2, partner bit, v0
        0x3404_0482,                                         // v_lshlrev_b32 v2, 2, v2
        0xD760_0007,
        256 | (128 + 43) << 9, // v_readlane_b32 s7, v0, 43
        0x7D88_0000,           // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A,           // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0005,           // s_cbranch_execz to the restore
        0xD8D8_0000,
        0x0300_0002, // ds_read_b32 v3, v2
        0x4A06_0607, // v_add_nc_u32 v3, s7, v3
        0xE070_1000,
        0x8002_0301, // buffer_store_dword v3, v1, s[8:11], 0 offen
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ];
    let mut native = native_of(&body, true, threads);
    let StageIo::Cs(ref mut cs) = native.io else {
        unreachable!()
    };
    cs.rsrc2 |= 1 << 15;
    assert!(cs.lds_bytes() >= threads * 4);
    native
}

#[test]
fn paired_lanes_share_lds_and_keep_divergent_stores_masked() {
    let _one = super::super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let host = device();
    for threads in [64, 128] {
        let shader = paired_fixture(&shared(threads));
        let Some(host) = &host else { continue };
        for limit in [17, 47, 83, 128] {
            let expected: Vec<u32> = (0..=threads)
                .map(|t| {
                    if t < threads && t < limit {
                        (t ^ (threads / 2)) + (t & !63) + 43
                    } else {
                        0
                    }
                })
                .collect();
            assert_eq!(
                readback(host, &shader, limit, threads + 1),
                expected,
                "{threads} threads, limit {limit}"
            );
        }
        eprintln!("paired shared memory passed GPU readback for {threads} threads");
    }
}

#[test]
fn paired_retry_selects_only_the_refused_width() {
    let mut native = native(&[
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBF86_0006, // s_cbranch_vccz to end
        0xD834_0000,
        0x0000_0001, // ds_write_b32 v1, v0
        0xD8D8_0000,
        0x0300_0001, // ds_read_b32 v3, v1
        0xE070_1000,
        0x8002_0301,
    ]);
    let StageIo::Cs(ref mut cs) = native.io else {
        unreachable!()
    };
    cs.rsrc2 |= 1 << 15;
    let bytes = super::super::container::build(&[]);
    let c = crate::Container::parse(&bytes).unwrap();
    let p = ir::build(&native.code, native.wave64, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let emit = |width, on| {
        let was = lower::TEST_HOST_WAVE.replace(Some(width));
        let result = super::super::emit_with_paired(&c, &native, &p, &st, &io, &res, on);
        lower::TEST_HOST_WAVE.set(was);
        result.unwrap()
    };
    let bounds = |s: &HostShader| {
        let c = crate::Container::parse(&s.dxil).unwrap();
        let psv = c.part(crate::FourCC::PSV0).unwrap().data;
        [20, 24, 40].map(|at| u32::from_le_bytes(psv[at..at + 4].try_into().unwrap()))
    };
    assert_eq!(bounds(&emit(32, false)), [64, 64, 64]);
    assert_eq!(bounds(&emit(32, true)), [32, 32, 32]);
    let native = emit(64, false);
    assert_eq!(bounds(&native), [64, 64, 64]);
    assert_eq!(emit(64, true), native);
}

#[test]
fn paired_lane_gather_and_first_active_lane_use_both_registers() {
    let native = native(&[
        0x3402_0082, // v_lshlrev_b32 v1, 2, v0
        0x3A04_00A0, // v_xor_b32 v2, 32, v0
        0x3404_0482, // v_lshlrev_b32 v2, 2, v2
        0xDACC_0000,
        0x0300_0002, // ds_bpermute_b32 v3, v2, v0
        0x7D82_0000, // v_cmp_lt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_0004, // s_cbranch_execz to restore
        0x7E0E_0500, // v_readfirstlane_b32 s7, v0
        0x4A06_0607, // v_add_nc_u32 v3, s7, v3
        0xE070_1000,
        0x8002_0301,
        0xBEFE_040C,
    ]);
    let shader = paired_fixture(&native);
    let _one = super::super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let Some(host) = device() else { return };
    for threshold in [10, 31, 54, 63] {
        let expected: Vec<_> = (0..64)
            .map(|t| {
                if t > threshold {
                    (t ^ 32) + threshold + 1
                } else {
                    0
                }
            })
            .collect();
        assert_eq!(
            readback(&host, &shader, threshold, 64),
            expected,
            "lanes above {threshold}"
        );
    }
    eprintln!("paired lane gather and first active lane passed GPU readback");
}
