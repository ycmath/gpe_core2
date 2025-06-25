# Step 4: 최종 벤치마크 및 사용 가이드

import json
import time
import pandas as pd
import matplotlib.pyplot as plt
from gpe_core.encoder import GPEEncoder
from gpe_core.decoder import GPEDecoder

print("=== GPE 최적화 통합 완료 ===\n")

# ===== 1. 종합 벤치마크 =====
print("1. 종합 성능 벤치마크\n")

def benchmark_gpe(data, name):
    """GPE 인코딩/디코딩 벤치마크"""
    results = {}
    
    # 원본 크기
    json_size = len(json.dumps(data))
    results['data_size'] = json_size
    
    # JSON 기준
    t0 = time.time()
    json_str = json.dumps(data)
    json_enc_time = time.time() - t0
    
    t0 = time.time()
    json.loads(json_str)
    json_dec_time = time.time() - t0
    
    results['json_time'] = (json_enc_time + json_dec_time) * 1000
    
    # GPE 최적화 없이
    encoder_no_opt = GPEEncoder(include_fallback=False, enable_optimization=False)
    decoder = GPEDecoder()
    
    t0 = time.time()
    payload_no_opt = encoder_no_opt.encode(data)
    enc_time_no_opt = time.time() - t0
    
    t0 = time.time()
    decoded_no_opt = decoder.decode(payload_no_opt)
    dec_time_no_opt = time.time() - t0
    
    results['gpe_no_opt_time'] = (enc_time_no_opt + dec_time_no_opt) * 1000
    results['gpe_no_opt_size'] = len(json.dumps(payload_no_opt.generative_payload))
    results['gpe_no_opt_valid'] = decoded_no_opt == data
    
    # GPE 최적화 포함
    encoder_opt = GPEEncoder(
        include_fallback=False,
        enable_optimization=True,
        optimization_config={'constant_threshold': 3, 'range_min_length': 5}
    )
    
    t0 = time.time()
    payload_opt, analysis = encoder_opt.encode_with_analysis(data)
    enc_time_opt = time.time() - t0
    
    t0 = time.time()
    decoded_opt = decoder.decode(payload_opt)
    dec_time_opt = time.time() - t0
    
    results['gpe_opt_time'] = (enc_time_opt + dec_time_opt) * 1000
    results['gpe_opt_size'] = len(json.dumps(payload_opt.generative_payload))
    results['gpe_opt_valid'] = decoded_opt == data
    results['optimization'] = analysis.get('optimization', {})
    
    # 압축률
    results['gpe_no_opt_compression'] = (1 - results['gpe_no_opt_size'] / json_size) * 100
    results['gpe_opt_compression'] = (1 - results['gpe_opt_size'] / json_size) * 100
    
    return results

# 다양한 패턴의 벤치마크 데이터
benchmark_cases = [
    {
        "name": "높은 반복 (90%)",
        "data": {"items": [{"type": "A", "status": "active"}] * 90 + 
                         [{"type": "B", "status": "inactive"}] * 10}
    },
    {
        "name": "중간 반복 (50%)",
        "data": {"values": [1] * 50 + [2] * 30 + [3] * 20}
    },
    {
        "name": "연속 패턴",
        "data": {"numbers": list(range(100)), 
                "ids": [f"ID_{i:04d}" for i in range(50)]}
    },
    {
        "name": "구조적 반복",
        "data": [{"id": i, "name": f"User {i}", "role": "user", 
                 "active": True, "score": i % 10} for i in range(100)]
    },
    {
        "name": "복합 패턴",
        "data": {
            "config": {"version": "1.0", "enabled": True},
            "users": [{"id": i, "type": "user"} for i in range(50)],
            "items": [{"id": i, "type": "item"} for i in range(50)],
            "flags": [True] * 80 + [False] * 20
        }
    }
]

# 벤치마크 실행
results_df = pd.DataFrame()

for case in benchmark_cases:
    print(f"테스트: {case['name']}")
    result = benchmark_gpe(case['data'], case['name'])
    result['name'] = case['name']
    
    # 출력
    print(f"  데이터 크기: {result['data_size']} bytes")
    print(f"  JSON 시간: {result['json_time']:.2f} ms")
    print(f"  GPE (최적화 X): {result['gpe_no_opt_time']:.2f} ms, "
          f"크기 {result['gpe_no_opt_size']} ({result['gpe_no_opt_compression']:.1f}%)")
    print(f"  GPE (최적화 O): {result['gpe_opt_time']:.2f} ms, "
          f"크기 {result['gpe_opt_size']} ({result['gpe_opt_compression']:.1f}%)")
    
    if result['optimization']:
        opt = result['optimization']
        print(f"  최적화 효과: {opt['original_rules']} → {opt['optimized_rules']} 규칙 "
              f"({(1-opt['optimized_rules']/opt['original_rules'])*100:.1f}% 감소)")
    print()
    
    results_df = pd.concat([results_df, pd.DataFrame([result])], ignore_index=True)

# ===== 2. 시각화 =====
print("\n2. 성능 시각화\n")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 압축률 비교
ax = axes[0, 0]
x = range(len(benchmark_cases))
ax.bar([i-0.2 for i in x], results_df['gpe_no_opt_compression'], 
       width=0.4, label='GPE (최적화 X)', alpha=0.7)
ax.bar([i+0.2 for i in x], results_df['gpe_opt_compression'], 
       width=0.4, label='GPE (최적화 O)', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels([c['name'] for c in benchmark_cases], rotation=45, ha='right')
ax.set_ylabel('압축률 (%)')
ax.set_title('압축률 비교')
ax.legend()
ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)

# 처리 시간 비교
ax = axes[0, 1]
ax.plot(x, results_df['json_time'], 'o-', label='JSON', linewidth=2)
ax.plot(x, results_df['gpe_no_opt_time'], 's-', label='GPE (최적화 X)', linewidth=2)
ax.plot(x, results_df['gpe_opt_time'], '^-', label='GPE (최적화 O)', linewidth=2)
ax.set_xticks(x)
ax.set_xticklabels([c['name'] for c in benchmark_cases], rotation=45, ha='right')
ax.set_ylabel('처리 시간 (ms)')
ax.set_yscale('log')
ax.set_title('처리 시간 비교')
ax.legend()

# 규칙 감소율
ax = axes[1, 0]
reduction_rates = []
for _, row in results_df.iterrows():
    if row['optimization'] and row['optimization'].get('original_rules', 0) > 0:
        rate = (1 - row['optimization']['optimized_rules'] / 
                row['optimization']['original_rules']) * 100
    else:
        rate = 0
    reduction_rates.append(rate)

ax.bar(x, reduction_rates, alpha=0.7, color='green')
ax.set_xticks(x)
ax.set_xticklabels([c['name'] for c in benchmark_cases], rotation=45, ha='right')
ax.set_ylabel('규칙 감소율 (%)')
ax.set_title('최적화로 인한 규칙 감소')

# 최적화 타입 분포
ax = axes[1, 1]
opt_types = {'constants': 0, 'ranges': 0, 'lists': 0, 'templates': 0}
for _, row in results_df.iterrows():
    if row['optimization']:
        opt = row['optimization']
        opt_types['constants'] += opt.get('constants_found', 0)
        opt_types['ranges'] += opt.get('ranges_found', 0)
        opt_types['lists'] += opt.get('lists_compressed', 0)
        opt_types['templates'] += opt.get('templates_found', 0)

ax.pie(opt_types.values(), labels=opt_types.keys(), autopct='%1.1f%%')
ax.set_title('최적화 타입 분포')

plt.tight_layout()
plt.savefig('/content/gpe_optimization_results.png', dpi=150)
print("✅ 성능 그래프 저장: /content/gpe_optimization_results.png")

# ===== 3. 실제 사용 예시 =====
print("\n3. 실제 사용 예시\n")

# API 응답 데이터 예시
api_response = {
    "status": "success",
    "timestamp": "2024-01-01T00:00:00Z",
    "data": {
        "users": [
            {
                "id": i,
                "username": f"user_{i}",
                "email": f"user_{i}@example.com",
                "profile": {
                    "firstName": f"First{i}",
                    "lastName": f"Last{i}",
                    "avatar": "/default-avatar.png",
                    "preferences": {
                        "theme": "dark",
                        "language": "en",
                        "notifications": True
                    }
                },
                "status": "active",
                "role": "user",
                "createdAt": "2024-01-01T00:00:00Z"
            }
            for i in range(100)
        ],
        "pagination": {
            "page": 1,
            "perPage": 100,
            "total": 100,
            "totalPages": 1
        }
    }
}

print("실제 API 응답 데이터 압축:")

# 표준 GPE
encoder_standard = GPEEncoder(include_fallback=False, enable_optimization=False)
payload_standard = encoder_standard.encode(api_response)

# 최적화 GPE
encoder_optimized = GPEEncoder(
    include_fallback=False,
    enable_optimization=True,
    optimization_config={
        'constant_threshold': 3,
        'template_threshold': 0.8,
        'range_min_length': 5
    }
)
payload_optimized, analysis = encoder_optimized.encode_with_analysis(api_response)

# 결과 비교
original_size = len(json.dumps(api_response))
standard_size = len(json.dumps(payload_standard.generative_payload))
optimized_size = len(json.dumps(payload_optimized.generative_payload))

print(f"  원본 크기: {original_size:,} bytes")
print(f"  표준 GPE: {standard_size:,} bytes ({(1-standard_size/original_size)*100:.1f}% 압축)")
print(f"  최적화 GPE: {optimized_size:,} bytes ({(1-optimized_size/original_size)*100:.1f}% 압축)")
print(f"  최적화 개선: {(1-optimized_size/standard_size)*100:.1f}%")

# ===== 4. 사용 가이드 =====
print("\n4. GPE 최적화 사용 가이드\n")

usage_guide = """
# 기본 사용법
from gpe_core.encoder import GPEEncoder
from gpe_core.decoder import GPEDecoder

# 1. 기본 인코딩/디코딩
encoder = GPEEncoder(enable_optimization=True)
decoder = GPEDecoder()

data = {"your": "data"}
payload = encoder.encode(data)
decoded = decoder.decode(payload)

# 2. 커스텀 최적화 설정
encoder = GPEEncoder(
    include_fallback=False,  # 폴백 JSON 제외
    enable_optimization=True,
    optimization_config={
        'constant_threshold': 5,    # 5번 이상 반복시 상수화
        'template_threshold': 0.7,  # 70
