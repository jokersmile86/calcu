"""
=========================================================
프로젝트 이름: Mini NPU Simulator (MAC 연산 기반 패턴 판별기)
설명: 필터와 패턴을 위치별로 곱하고 누적해서 더하는 MAC(Multiply-Accumulate)
      연산을 반복문으로 직접 구현하고, 3x3 ~ 25x25 패턴을 판별/성능 측정한다.
제약: Python 표준 라이브러리(json, time, math, re, os, sys)만 사용
=========================================================
"""

import json
import math
import os
import re
import sys
import time

# ---------------------------------------------------------
# 설정값
# ---------------------------------------------------------
EPSILON = 1e-9          # 점수 비교 허용오차 (|a-b| < EPSILON 이면 동점)
REPEAT = 10             # 성능 측정 반복 횟수
CROSS = "Cross"         # 표준 라벨 1
X_LABEL = "X"           # 표준 라벨 2
UNDECIDED = "UNDECIDED"
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

# 라벨 정규화 표: expected('+', 'x')와 filter 키('cross', 'x')를 모두 처리
LABEL_MAP = {"+": CROSS, "cross": CROSS, "x": X_LABEL}

GEN_MIN_SIZE = 3        # 패턴 생성기 최소 크기
GEN_MAX_SIZE = 100      # 패턴 생성기 최대 크기 (입력 실수로 인한 과도한 연산 방지)
PERF_SIZES = (3, 5, 13, 25)   # 성능 분석 기본 크기


# =========================================================
# 1. 데이터 구조
# =========================================================
class Matrix:
    """n x n 크기의 2차원 숫자 배열 (패턴/필터 공용)"""

    def __init__(self, size):
        self.size = size
        self.rows = [[0.0] * size for _ in range(size)]

    def get(self, r, c):
        """(r, c) 위치의 값을 읽어온다."""
        return self.rows[r][c]

    def set(self, r, c, value):
        """(r, c) 위치에 값을 저장한다."""
        self.rows[r][c] = value

    @classmethod
    def from_rows(cls, rows):
        """
        2차원 리스트로 Matrix 생성. 아래의 경우 ValueError(사유 메시지) 발생
        - 리스트가 아니거나 비어 있음 / 정사각형이 아님 / 숫자가 아닌 값(NaN, inf 포함)
        """
        if not isinstance(rows, list) or len(rows) == 0:
            raise ValueError("2차원 배열(리스트)이 아닙니다.")
        n = len(rows)
        matrix = cls(n)
        for i, row in enumerate(rows):
            if not isinstance(row, list):
                raise ValueError(f"{i + 1}행이 리스트가 아닙니다.")
            if len(row) != n:
                raise ValueError(f"정사각형이 아닙니다. ({i + 1}행 길이 {len(row)}, 기대 {n})")
            for j, value in enumerate(row):
                # bool은 int의 하위 타입이므로 별도로 제외
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError(f"숫자가 아닌 값이 있습니다. ({i + 1}행 {j + 1}열: {value!r})")
                # 매우 큰 정수는 float 변환 과정에서 OverflowError가 날 수 있다.
                try:
                    numeric_value = float(value)
                except (OverflowError, ValueError):
                    raise ValueError(f"표현 가능한 숫자 범위를 벗어났습니다. ({i + 1}행 {j + 1}열)")
                if not math.isfinite(numeric_value):
                    raise ValueError(f"유한한 숫자가 아닙니다. ({i + 1}행 {j + 1}열)")
                matrix.set(i, j, numeric_value)
        return matrix

    def show(self, indent="  "):
        for row in self.rows:
            print(indent + " ".join(f"{v:g}" for v in row))


# =========================================================
# 1-2. 보너스: 패턴 생성기 (N x N 십자가 / X)
# =========================================================
def _check_generator_size(n):
    """생성 크기 검증. 정수이고 GEN_MIN_SIZE~GEN_MAX_SIZE 범위여야 한다."""
    if isinstance(n, bool) or not isinstance(n, int):
        raise ValueError("크기 N 은 정수여야 합니다.")
    if not GEN_MIN_SIZE <= n <= GEN_MAX_SIZE:
        raise ValueError(
            f"크기 N 은 {GEN_MIN_SIZE} 이상 {GEN_MAX_SIZE} 이하여야 합니다. (입력 {n})"
        )


def generate_cross(n):
    """
    N x N 십자가(Cross) 패턴 생성: 가운데 행과 가운데 열을 1, 나머지를 0으로 채운다.
    N 이 짝수이면 가운데가 두 줄이므로 가운데 두 행/열을 모두 1로 채운다.
    """
    _check_generator_size(n)
    mids = {(n - 1) // 2, n // 2}      # 홀수 N: 한 줄 / 짝수 N: 두 줄
    matrix = Matrix(n)
    for r in range(n):
        for c in range(n):
            if r in mids or c in mids:
                matrix.set(r, c, 1.0)
    return matrix


def generate_x(n):
    """N x N X 패턴 생성: 주 대각선(r == c)과 반대 대각선(r + c == N-1)을 1로 채운다."""
    _check_generator_size(n)
    matrix = Matrix(n)
    for i in range(n):
        matrix.set(i, i, 1.0)
        matrix.set(i, n - 1 - i, 1.0)
    return matrix


# =========================================================
# 2. 라벨 정규화 / MAC 연산 / 판정 / 측정
# =========================================================
def normalize_label(raw):
    """
    다양한 라벨 표기를 표준 라벨(Cross, X)로 통일한다.
    - expected: '+' -> Cross, 'x' -> X
    - filter 키: 'cross' -> Cross, 'x' -> X
    알 수 없는 값이면 ValueError
    """
    if not isinstance(raw, str):
        raise ValueError(f"라벨이 문자열이 아닙니다: {raw!r}")
    key = raw.strip().lower()
    if key not in LABEL_MAP:
        raise ValueError(f"알 수 없는 라벨입니다: {raw!r}")
    return LABEL_MAP[key]


def mac(pattern, filt):
    """
    MAC(Multiply-Accumulate) 연산: 같은 위치의 값끼리 곱하고 모두 더한다.
    외부 라이브러리 없이 이중 반복문으로 구현 (연산 횟수 = N^2)
    """
    if pattern.size != filt.size:
        raise ValueError(f"크기 불일치: 패턴 {pattern.size}x{pattern.size}, 필터 {filt.size}x{filt.size}")
    total = 0.0
    for i in range(pattern.size):
        for j in range(pattern.size):
            product = pattern.get(i, j) * filt.get(i, j)
            if not math.isfinite(product):
                raise ValueError(f"MAC 곱셈 결과가 유한하지 않습니다. ({i + 1}행 {j + 1}열)")
            total += product
            if not math.isfinite(total):
                raise ValueError(f"MAC 누적 결과가 유한하지 않습니다. ({i + 1}행 {j + 1}열)")
    return total


def judge(score_a, score_b, label_a, label_b):
    """
    두 점수를 허용오차(EPSILON) 기반으로 비교해 판정한다.
    float 연산은 0.9 와 0.8999999999999999 처럼 미세한 오차가 생기므로 == 를 쓰지 않는다.
    """
    if not math.isfinite(score_a) or not math.isfinite(score_b):
        raise ValueError("판정 점수는 유한한 숫자여야 합니다.")
    if abs(score_a - score_b) < EPSILON:
        return UNDECIDED
    return label_a if score_a > score_b else label_b


def measure_mac(pattern, filt, repeat=REPEAT):
    """
    MAC 연산 구간만 repeat 회 측정해 (평균 시간 ms, 점수)를 반환한다.
    입력/출력/파일 읽기 시간은 포함하지 않는다.
    """
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat <= 0:
        raise ValueError("측정 반복 횟수는 1 이상의 정수여야 합니다.")
    elapsed = []
    score = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        score = mac(pattern, filt)
        elapsed.append((time.perf_counter() - start) * 1000)
    return sum(elapsed) / len(elapsed), score


# =========================================================
# 3. 모드 1: 사용자 입력 (3x3)
# =========================================================
def read_row(n, prompt="  > "):
    """한 줄을 입력받아 n개의 숫자 리스트로 변환. 형식 오류면 안내 후 그 줄만 재입력"""
    while True:
        line = input(prompt)
        parts = line.split()
        if len(parts) != n:
            print(f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요. (현재 {len(parts)}개)")
            continue
        try:
            values = [float(p) for p in parts]
        except ValueError:
            print(f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요. (숫자가 아닌 값 포함)")
            continue
        if not all(math.isfinite(v) for v in values):
            print(f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요. (nan/inf 불가)")
            continue
        return values


def read_matrix(title, n=3):
    """n줄을 입력받아 Matrix로 저장"""
    print(f"{title} ({n}줄 입력, 공백 구분)")
    rows = [read_row(n) for _ in range(n)]
    return Matrix.from_rows(rows)


def print_section(title):
    print("\n#---------------------------------------")
    print(f"# {title}")
    print("#---------------------------------------")


def print_perf_table(entries):
    """entries: [(크기 N, 평균 ms) ...] 형태로 성능 표 출력"""
    print(f"{'크기':<10}{'평균 시간(ms)':<18}{'연산 횟수(N²)'}")
    print("-" * 44)
    for n, avg_ms in entries:
        label = f"{n}×{n}"
        value = "데이터 없음" if avg_ms is None else f"{avg_ms:.4f}"
        print(f"{label:<10}{value:<18}{n * n}")


def print_mac_result(pattern, filter_a, filter_b, label_a="A", label_b="B"):
    """
    두 필터와 패턴의 MAC 점수/연산 시간/판정을 출력하고 (판정, 평균 시간 ms)를 반환한다.
    모드 1(A/B)과 모드 3(Cross/X)이 함께 사용한다.
    """
    try:
        avg_a, score_a = measure_mac(pattern, filter_a)
        avg_b, score_b = measure_mac(pattern, filter_b)
        avg_ms = (avg_a + avg_b) / 2
        verdict = judge(score_a, score_b, label_a, label_b)
    except ValueError as e:
        print_section("[3] MAC 연산 오류")
        print(f"연산을 완료할 수 없습니다: {e}")
        return UNDECIDED, None

    if verdict == UNDECIDED:
        print_section("[3] MAC 결과 (판정 불가)")
    else:
        print_section("[3] MAC 결과")
    print(f"{label_a} 점수: {score_a!r}")
    print(f"{label_b} 점수: {score_b!r}")
    print(f"연산 시간(평균/{REPEAT}회): {avg_ms:.4f} ms")
    if verdict == UNDECIDED:
        print(f"판정: 판정 불가 (|{label_a}-{label_b}| < {EPSILON:g})")
    else:
        print(f"판정: {verdict}")
    return verdict, avg_ms


def run_user_mode():
    """모드 1: 필터 A, B와 패턴을 입력받아 MAC 점수/판정/연산 시간 출력"""
    print_section("[1] 필터 입력")
    filter_a = read_matrix("필터 A")
    print()
    filter_b = read_matrix("필터 B")

    print("\n✓ 저장 확인 - 필터 A")
    filter_a.show()
    print("✓ 저장 확인 - 필터 B")
    filter_b.show()

    print_section("[2] 패턴 입력")
    pattern = read_matrix("패턴")
    print("✓ 저장 확인 - 패턴")
    pattern.show()

    _, avg_ms = print_mac_result(pattern, filter_a, filter_b)

    print_section(f"[4] 성능 분석 (평균/{REPEAT}회)")
    print_perf_table([(3, avg_ms)])


# =========================================================
# 3-2. 모드 3 (보너스): 패턴 생성기
# =========================================================
def read_size():
    """생성할 크기 N 을 입력받는다. 정수가 아니거나 범위를 벗어나면 안내 후 재입력"""
    while True:
        raw = input(f"생성할 크기 N 입력 ({GEN_MIN_SIZE}~{GEN_MAX_SIZE}): ").strip()
        try:
            n = int(raw)
        except ValueError:
            print(f"입력 형식 오류: {GEN_MIN_SIZE} 이상 {GEN_MAX_SIZE} 이하의 정수를 입력하세요.")
            continue
        if not GEN_MIN_SIZE <= n <= GEN_MAX_SIZE:
            print(f"입력 형식 오류: {GEN_MIN_SIZE} 이상 {GEN_MAX_SIZE} 이하의 정수를 입력하세요.")
            continue
        return n


def read_pattern_choice():
    """판별할 패턴 종류를 선택받는다. 반환: 표준 라벨(Cross 또는 X)"""
    while True:
        choice = input("판별할 패턴 선택 (1: Cross, 2: X): ").strip()
        if choice == "1":
            return CROSS
        if choice == "2":
            return X_LABEL
        print("⚠️ 1 또는 2 중에서 선택해주세요.")


def run_generator_mode():
    """
    모드 3: N x N Cross/X 필터를 자동 생성하고, 생성한 패턴 1개를 MAC 판정한 뒤
    생성기로 만든 패턴으로 성능 분석까지 수행한다. (모드 1의 판정/출력 함수를 재사용)
    """
    print_section("[1] 패턴 생성")
    n = read_size()
    filter_cross, filter_x = generate_cross(n), generate_x(n)
    print(f"\n✓ 생성 완료 - Cross {n}x{n}")
    filter_cross.show()
    print(f"✓ 생성 완료 - X {n}x{n}")
    filter_x.show()

    print_section("[2] 패턴 선택")
    expected = read_pattern_choice()
    pattern = generate_cross(n) if expected == CROSS else generate_x(n)
    print(f"✓ 입력 패턴: 생성된 {expected} {n}x{n}")

    verdict, _ = print_mac_result(pattern, filter_cross, filter_x, CROSS, X_LABEL)
    print(f"expected: {expected} | {'PASS' if verdict == expected else 'FAIL'}")

    print_section(f"[4] 성능 분석 (평균/{REPEAT}회, 생성기 패턴 사용)")
    perf = []
    for size in sorted(set(PERF_SIZES) | {n}):
        try:
            avg_ms, _ = measure_mac(generate_cross(size), generate_x(size))
        except ValueError:
            avg_ms = None
        perf.append((size, avg_ms))
    print_perf_table(perf)


# =========================================================
# 4. 모드 2: data.json 분석
# =========================================================
FILTER_KEY_RE = re.compile(r"^size_(\d+)$")
PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(\d+)$")


def load_json_file(path):
    """JSON 파일을 읽어 최상위 스키마(filters/patterns)까지 검사. 문제 시 ValueError(사유)"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"파일을 찾을 수 없습니다: {path}")
    except PermissionError:
        raise ValueError(f"파일 읽기 권한이 없습니다: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON 형식이 올바르지 않습니다: {e}")
    except (OSError, UnicodeDecodeError) as e:
        raise ValueError(f"파일을 읽을 수 없습니다: {e}")

    if not isinstance(data, dict):
        raise ValueError("최상위 데이터가 객체(dict)가 아닙니다.")
    for key in ("filters", "patterns"):
        if not isinstance(data.get(key), dict):
            raise ValueError(f"'{key}' 항목이 없거나 객체(dict)가 아닙니다.")
    return data


def build_filter(size, raw_filter):
    """
    size_N 필터 하나를 {Cross: Matrix, X: Matrix}로 변환 (라벨 정규화 적용)
    문제가 있으면 ValueError(사유)
    """
    if not isinstance(raw_filter, dict):
        raise ValueError("필터 항목이 객체(dict)가 아닙니다.")
    result = {}
    for raw_label, rows in raw_filter.items():
        label = normalize_label(raw_label)
        if label in result:
            raise ValueError(f"라벨 중복: {label}")
        matrix = Matrix.from_rows(rows)
        if matrix.size != size:
            raise ValueError(f"{label} 필터 크기 {matrix.size}x{matrix.size} (기대 {size}x{size})")
        result[label] = matrix
    for label in (CROSS, X_LABEL):
        if label not in result:
            raise ValueError(f"{label} 필터가 없습니다.")
    return result


def load_filters(raw_filters):
    """
    모든 필터를 로드. 반환: (filters {N: {Cross, X}}, errors {키: 사유})
    필터 하나가 잘못돼도 나머지는 계속 로드한다.
    """
    filters, errors = {}, {}
    for key, raw in raw_filters.items():
        m = FILTER_KEY_RE.match(key)
        if not m:
            errors[key] = "필터 키 형식이 size_N 이 아닙니다."
            continue
        size = int(m.group(1))
        try:
            filters[size] = build_filter(size, raw)
        except ValueError as e:
            errors[key] = str(e)
    return filters, errors


def pattern_sort_key(key):
    """size_N_idx 키를 (N, idx) 숫자 순으로 정렬 (형식이 틀린 키는 뒤로)"""
    m = PATTERN_KEY_RE.match(key)
    if m:
        return (0, int(m.group(1)), int(m.group(2)), key)
    return (1, 0, 0, key)


def filter_sort_key(key):
    """size_N 키를 N 숫자 순으로 정렬 (형식이 틀린 키는 뒤로)"""
    m = FILTER_KEY_RE.match(key)
    return (0, int(m.group(1)), key) if m else (1, 0, key)


def evaluate_pattern(key, entry, filters, filter_errors):
    """
    패턴 1건을 검증/판정해 결과 dict를 반환한다. 어떤 문제가 있어도 예외를 밖으로
    내보내지 않고 FAIL(사유)로 기록한다. (프로그램 비정상 종료 방지)
    반환: {key, ok, reason, cross, x, verdict, expected, pattern, size}
    """
    result = {"key": key, "ok": False, "reason": "", "cross": None, "x": None,
              "verdict": None, "expected": None, "pattern": None, "size": None}
    try:
        m = PATTERN_KEY_RE.match(key)
        if not m:
            raise ValueError("패턴 키 형식이 size_N_idx 가 아닙니다.")
        size = int(m.group(1))
        result["size"] = size

        if not isinstance(entry, dict):
            raise ValueError("패턴 항목이 객체(dict)가 아닙니다.")
        if "input" not in entry or "expected" not in entry:
            raise ValueError("input 또는 expected 키가 없습니다.")

        filter_key = f"size_{size}"
        if filter_key in filter_errors:
            raise ValueError(f"{filter_key} 필터 로드 실패: {filter_errors[filter_key]}")
        if size not in filters:
            raise ValueError(f"{filter_key} 필터가 data.json 에 없습니다.")

        pattern = Matrix.from_rows(entry["input"])
        if pattern.size != size:
            raise ValueError(f"패턴 크기 {pattern.size}x{pattern.size} 와 필터 크기 {size}x{size} 불일치")
        expected = normalize_label(entry["expected"])

        score_cross = mac(pattern, filters[size][CROSS])
        score_x = mac(pattern, filters[size][X_LABEL])
        verdict = judge(score_cross, score_x, CROSS, X_LABEL)

        result.update(cross=score_cross, x=score_x, verdict=verdict,
                      expected=expected, pattern=pattern)
        if verdict == UNDECIDED:
            result["reason"] = f"동점(UNDECIDED) 처리 규칙에 따라 FAIL (|Cross-X| < {EPSILON:g})"
        elif verdict != expected:
            result["reason"] = f"판정 {verdict} 가 expected {expected} 와 다름"
        else:
            result["ok"] = True
    except ValueError as e:
        result["reason"] = str(e)
    return result


def print_case(result):
    print(f"--- {result['key']} ---")
    if result["cross"] is None:
        print(f"✗ FAIL: {result['reason']}")
        return
    print(f"Cross 점수: {result['cross']!r}")
    print(f"X 점수: {result['x']!r}")
    status = "PASS" if result["ok"] else "FAIL"
    note = "" if result["ok"] else " (동점 규칙)" if result["verdict"] == UNDECIDED else ""
    print(f"판정: {result['verdict']} | expected: {result['expected']} | {status}{note}")


def run_data_mode(path=DATA_FILE):
    """모드 2: data.json 의 모든 패턴을 일괄 판정하고 성능 분석/결과 요약 출력"""
    try:
        data = load_json_file(path)
    except ValueError as e:
        print(f"⚠️ data.json 을 불러올 수 없습니다: {e}")
        return

    # [1] 필터 로드
    print_section("[1] 필터 로드")
    filters, filter_errors = load_filters(data["filters"])
    for key in sorted(data["filters"], key=filter_sort_key):
        if key in filter_errors:
            print(f"✗ {key} 필터 로드 실패: {filter_errors[key]}")
        else:
            print(f"✓ {key:<8} 필터 로드 완료 ({CROSS}, {X_LABEL})")

    # [2] 패턴 분석
    print_section("[2] 패턴 분석 (라벨 정규화 적용)")
    results = []
    for key in sorted(data["patterns"], key=pattern_sort_key):
        result = evaluate_pattern(key, data["patterns"][key], filters, filter_errors)
        print_case(result)
        results.append(result)

    # [3] 성능 분석 (3x3은 기본 예시, 나머지는 data.json의 필터/패턴 사용)
    print_section(f"[3] 성능 분석 (평균/{REPEAT}회)")
    print("측정 기준: 크기별 첫 번째 유효 패턴 × Cross 필터의 MAC 연산 시간")
    samples = {3: (generate_cross(3), generate_x(3))}
    for r in results:
        if r["pattern"] is not None and r["size"] not in samples:
            samples[r["size"]] = (r["pattern"], filters[r["size"]][CROSS])
    perf = []
    for n in sorted(set(PERF_SIZES) | set(samples)):
        if n in samples:
            try:
                avg_ms, _ = measure_mac(samples[n][0], samples[n][1])
            except ValueError:
                avg_ms = None
            perf.append((n, avg_ms))
        else:
            perf.append((n, None))
    print_perf_table(perf)

    # [4] 결과 요약
    print_section("[4] 결과 요약")
    failed = [r for r in results if not r["ok"]]
    print(f"총 테스트: {len(results)}개")
    print(f"통과: {len(results) - len(failed)}개")
    print(f"실패: {len(failed)}개")
    if failed:
        print("\n실패 케이스:")
        for r in failed:
            print(f"- {r['key']}: {r['reason']}")
    print("\n(상세 원인 분석 및 복잡도 설명은 README.md 의 '결과 리포트' 섹션 참고)")


# =========================================================
# 5. 메인 메뉴
# =========================================================
def main():
    print("=== Mini NPU Simulator ===")
    try:
        while True:
            print("\n[모드 선택]\n")
            print("1. 사용자 입력 (3x3)")
            print("2. data.json 분석")
            print("3. 패턴 생성기 (Cross / X 자동 생성)")
            print("0. 종료")
            choice = input("선택: ").strip()
            if choice == "1":
                run_user_mode()
            elif choice == "2":
                run_data_mode()
            elif choice == "3":
                run_generator_mode()
            elif choice == "0":
                print("👋 프로그램을 종료합니다.")
                break
            else:
                print("⚠️ 0, 1, 2, 3 중에서 선택해주세요.")
    except (KeyboardInterrupt, EOFError):
        print("\n\n⚠️ 프로그램을 안전하게 종료합니다.")
        sys.exit(0)


if __name__ == "__main__":
    main()