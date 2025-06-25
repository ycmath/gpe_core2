# Changelog

## [1.1.0-alpha] - 2025-06-25
### Added
- **RuleOptimizer 레이어**  
  - `ConstantRule`, `RangeRule`, `CompactListRule`, `TemplateRule`
  - `RuleOptimizer.optimize_rules()` 및 `get_stats()` 통계 리턴
- **Encoder**
  - `enable_optimization` 플래그 (`False` 기본값)
  - v1.1 페이로드: flat-rule 배열 + `version: "gpe.v1.1"`
- **Decoder**
  - v1.1 (flat rules) ↔ v1.0 (seeds) 자동 식별
  - 새 OP 지원: `CONSTANT / RANGE / COMPACT_LIST`
  - `GPEDecodeError` 로 오류 유형 명확화
- **멀티-프로세스 디코더**
  - `GPEDecoderMP`, `GPEDecoderMP_ShMem`  
    v1.1 입력 시 단일-프로세스 fallback 처리
- **GPU 백엔드**
  - `GPEDecoderGPUStream` (ID remap GPU + CPU 조립)
  - `GPEDecoderGPUStreamFull` (Full GPU graph 조립)  
    - v1.1 입력은 자동 CPU-fallback
- **CLI**
  - `--backend` 선택지 확대: `gpu-stream`, `gpu-full`, `gpu-ray`
  - `gpe bench` — synthetic 레코드 encode/decode 벤치마크

### Changed
- `pyproject.toml` → `version = "1.1.0a0"`, CuPy 12.x 지원
- GPU 커널 컴파일 옵션에서 `-O3` 제거, `--std=c++11` 기본 추가
- `assemble_graph.*.cu` → `<stdint.h>` 포함, 불필요 주석 제거

### Fixed
- CuPy 12.x 환경에서 bytes → uint8 변환 오류 (`ValueError: invalid literal for int()`)
- NVRTC `-O3` / invalid option 에러
- Numba-비활성 워커가 dict 타입을 pickle-unsafe 로 전달하던 문제

### Performance
| 경로          | v1.0 baseline | v1.0(변경 후) | v1.1(flat) |
|---------------|--------------:|--------------:|-----------:|
| CPU-단일      | 1 × | 1 × | +3 % |
| **MP(2 proc)**| 1.0 × | **1.9 ×** | 1.1 × |
| **GPU-Stream**| 1.0 × | **2.7 ×** | CPU-fallback |
| 압축률(v1.1)  | - | - | **-42 % (payload size)** |

> Colab (T4, Python 3.11, NumPy 1.26, CuPy 12.3) 기준.

---

## [1.0.3] - 2025-04-02
*Bugfix release — 자세한 내역 생략*

# Changelog

## [1.1.0-alpha] - 2025-06-25
### Added
- **RuleOptimizer 레이어**  
  - `ConstantRule`, `RangeRule`, `CompactListRule`, `TemplateRule`
  - `RuleOptimizer.optimize_rules()` 및 `get_stats()` 통계 리턴
- **Encoder**
  - `enable_optimization` 플래그 (`False` 기본값)
  - v1.1 페이로드: flat-rule 배열 + `version: "gpe.v1.1"`
- **Decoder**
  - v1.1 (flat rules) ↔ v1.0 (seeds) 자동 식별
  - 새 OP 지원: `CONSTANT / RANGE / COMPACT_LIST`
  - `GPEDecodeError` 로 오류 유형 명확화
- **멀티-프로세스 디코더**
  - `GPEDecoderMP`, `GPEDecoderMP_ShMem`  
    v1.1 입력 시 단일-프로세스 fallback 처리
- **GPU 백엔드**
  - `GPEDecoderGPUStream` (ID remap GPU + CPU 조립)
  - `GPEDecoderGPUStreamFull` (Full GPU graph 조립)  
    - v1.1 입력은 자동 CPU-fallback
- **CLI**
  - `--backend` 선택지 확대: `gpu-stream`, `gpu-full`, `gpu-ray`
  - `gpe bench` — synthetic 레코드 encode/decode 벤치마크

### Changed
- `pyproject.toml` → `version = "1.1.0a0"`, CuPy 12.x 지원
- GPU 커널 컴파일 옵션에서 `-O3` 제거, `--std=c++11` 기본 추가
- `assemble_graph.*.cu` → `<stdint.h>` 포함, 불필요 주석 제거

### Fixed
- CuPy 12.x 환경에서 bytes → uint8 변환 오류 (`ValueError: invalid literal for int()`)
- NVRTC `-O3` / invalid option 에러
- Numba-비활성 워커가 dict 타입을 pickle-unsafe 로 전달하던 문제

### Performance
| 경로          | v1.0 baseline | v1.0(변경 후) | v1.1(flat) |
|---------------|--------------:|--------------:|-----------:|
| CPU-단일      | 1 × | 1 × | +3 % |
| **MP(2 proc)**| 1.0 × | **1.9 ×** | 1.1 × |
| **GPU-Stream**| 1.0 × | **2.7 ×** | CPU-fallback |
| 압축률(v1.1)  | - | - | **-42 % (payload size)** |

> Colab (T4, Python 3.11, NumPy 1.26, CuPy 12.3) 기준.

---

## [1.0.3] - 2025-04-02
*Bugfix release — 자세한 내역 생략*

