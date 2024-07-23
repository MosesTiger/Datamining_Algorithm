### 단순법(Simplex Algorithm) 직접 구현

#### 문제 정의
문제를 다음과 같이 정의해보겠습니다:

- **목적 함수**: minimize \( c^T x \)
- **제약 조건**: \( A x \le b \)

#### 예제 문제
목적 함수: \( z = -3x_1 - x_2 \) minimize  
제약 조건:
\[
\begin{align*}
x_1 + 2x_2 &\le 4 \\
3x_1 + 2x_2 &\le 6 \\
x_1, x_2 &\ge 0
\end{align*}
\]

### 단순법 알고리즘 구현
단순법을 구현하기 위해 다음 단계를 따릅니다:

1. 표준 형식으로 변환 (표준형)
2. 초기 기본 해 설정
3. 단순법 반복
4. 종료 조건 확인

#### 표준형으로 변환
제약 조건을 표준형으로 변환하면 다음과 같습니다:
\[
\begin{align*}
x_1 + 2x_2 + s_1 &= 4 \\
3x_1 + 2x_2 + s_2 &= 6 \\
x_1, x_2, s_1, s_2 &\ge 0 \\
\end{align*}
\]
여기서 \( s_1, s_2 \)는 슬랙 변수입니다.

#### 초기 기본 해 설정
초기 기본 해를 설정합니다. \( x_1 = x_2 = 0 \), \( s_1 = 4 \), \( s_2 = 6 \)입니다.

#### 파이썬 코드
다음은 단순법 알고리즘을 구현한 예제입니다:

```python
import numpy as np

# 문제 정의
c = np.array([-3, -1, 0, 0])  # 목적 함수 계수
A = np.array([
    [

```python
import numpy as np

# 문제 정의
c = np.array([-3, -1, 0, 0])  # 목적 함수 계수
A = np.array([
    [1, 2, 1, 0], 
    [3, 2, 0, 1]
])  # 제약 조건 계수
b = np.array([4, 6])  # 제약 조건의 우변 값

# 초기 기본 해 설정
basic_vars = [2, 3]
non_basic_vars = [0, 1]
B = A[:, basic_vars]
N = A[:, non_basic_vars]
c_B = c[basic_vars]
c_N = c[non_basic_vars]

# 반복 시작
while True:
    # Step 1: B^-1 구하기
    B_inv = np.linalg.inv(B)

    # Step 2: 현재 기본 해 구하기
    x_B = np.dot(B_inv, b)

    # Step 3: 현재 목적 함수 값 계산
    z = np.dot(c_B, x_B)

    # Step 4: Reduced cost 계산
    y = np.dot(c_B, B_inv)
    reduced_cost = c_N - np.dot(y, N)

    # 종료 조건 확인
    if all(reduced_cost >= 0):
        break  # 최적해 도출

    # Step 5: 들어올 변수 선택 (가장 작은 reduced cost)
    entering = np.argmin(reduced_cost)
    entering_var = non_basic_vars[entering]

    # Step 6: 들어올 변수에 대한 방향 벡터 계산
    direction = np.dot(B_inv, A[:, entering_var])

    # Step 7: 나갈 변수 선택 (Bland's rule)
    ratios = np.divide(x_B, direction, out=np.full_like(x_B, np.inf), where=direction > 0)
    leaving = np.argmin(ratios)
    leaving_var = basic_vars[leaving]

    # 변수 교체
    basic_vars[leaving] = entering_var
    non_basic_vars[entering] = leaving_var

    B = A[:, basic_vars]
    N = A[:, non_basic_vars]
    c_B = c[basic_vars]
    c_N = c[non_basic_vars]

# 결과 출력
solution = np.zeros(len(c))
solution[basic_vars] = x_B
print("최적해:", solution[:2])
print("최적해의 값:", z)
```

### 요약
이 코드는 단순법 알고리즘을 사용하여 주어진 선형 계획법 문제를 해결합니다. 각 단계는 수학적으로 단순법을 수행하는 방법을 반영하며, 최적해를 찾을 때까지 반복합니다. 이 예제는 단순한 형태의 문제를 해결하기 위한 것이며, 실제 응용에서는 더 복잡한 기능과 예외 처리가 필요할 수 있습니다.
