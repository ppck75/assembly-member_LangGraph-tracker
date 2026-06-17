# 국회의원 의정활동 추적 AI 에이전트 계획서

## 1. 프로젝트 개요

이 프로젝트는 사용자가 국회의원 이름을 입력하면 해당 의원의 기본정보를 먼저 확인한 뒤, 발의법안, 본회의 표결 기록, 최근 뉴스 이슈를 병렬로 수집하고, 마지막에 공동발의 네트워크, 정당 다수 입장 일치도, 핵심 브리핑을 종합하는 LangGraph 기반 AI 에이전트이다.

주요 구현 파일은 `src/member_activity_workflow.py`이며, Streamlit 앱의 실행 진입점은 `app.py`이다. 실제 워크플로우는 `StateGraph(MemberActivityState)`로 구성되어 있고, 각 단계는 독립된 노드로 분리되어 있다. 서로의 산출물을 기다릴 필요가 없는 노드는 fan-out/fan-in 방식으로 병렬 실행된다.

사용 LLM 모델은 코드 기준 `gemini-2.5-flash-lite`로 고정되어 있다.

```python
DEFAULT_LLM_MODEL = "gemini-2.5-flash-lite"
```

## 2. 구현한 Tool 종류와 기능

이 프로젝트에서 tool은 LangGraph 노드와 외부 데이터 호출 기능을 포함한다. 각 tool은 LLM이 임의로 순서를 정하는 방식이 아니라, LangGraph의 정해진 edge와 의존 관계에 따라 실행된다. 독립적인 데이터 수집·분석 노드는 병렬 실행되고, 필요한 결과가 모두 모인 뒤 fan-in 노드에서 종합된다.

### 2.1 입력 정리 Tool

노드명: `normalize_user_request`

기능:
- 사용자가 입력한 의원명을 정규화한다.
- 조회 대상 대수, 표결 조회 범위, LLM 사용 여부, 뉴스 검색 여부 등 실행 옵션을 `MemberActivityState`에 저장한다.
- 이후 노드들이 같은 상태값을 공유할 수 있도록 초기 state를 구성한다.

### 2.2 의원 기본정보 조회 Tool

노드명: `get_member_info`

사용 API:
- 열린국회정보 `ALLNAMEMBER`

기능:
- 의원 이름으로 국회의원 기본정보를 조회한다.
- 정당, 선거구, 위원회, 재임 대수를 확인한다.
- 의원이 실제로 재임한 대수만 분석 대상으로 제한한다.
- 표결 API가 지원하는 재임 대수 중 초기 조회 대수와 추가 조회 가능 대수를 분리한다.

### 2.3 발의법안 조회 Tool

노드명: `search_member_bills`

사용 API:
- 열린국회정보 `nzmimeepazxkubdpn`

기능:
- 의원별 발의법안 목록을 조회한다.
- 대표발의와 공동발의를 분류한다.
- 발의법안 처리결과별 통계와 소관위원회별 통계를 계산한다.
- 공동발의 보강 스캔 옵션을 통해 공동발의 누락 가능성을 줄인다.
- 의원명, 대수, 조회 옵션 기준으로 발의법안 결과를 캐시한다.

### 2.4 입법 관심 분야 분석 Tool

노드명: `analyze_legislative_interests`

사용 데이터:
- 발의법안 목록
- 대표발의/공동발의 수
- 처리결과별 통계
- 소관위원회별 통계

기능:
- 발의법안 통계와 최근 발의 법안을 바탕으로 의원의 입법 관심 분야를 요약한다.
- LLM 사용이 꺼져 있거나 Gemini 호출이 실패하면 규칙 기반 fallback 요약을 사용한다.

### 2.5 공동발의 네트워크 분석 Tool

노드명: `analyze_cosponsor_network`

사용 데이터:
- 공동발의 법안 목록
- 공동발의자 정보
- 의원 정당 매칭 정보

기능:
- 공동발의 파트너와 협업 연결 수를 계산한다.
- 당내 협업, 당외 협업, 정당 미확인 협업을 구분한다.
- 협업 파트너 집중도와 협업 유형을 계산한다.
- LLM을 통해 공동발의 네트워크 해석을 생성하거나, 실패 시 규칙 기반 분석으로 대체한다.

### 2.6 표결 상세 조회 Tool

노드명: `get_all_member_votes`

사용 API:
- 열린국회정보 `ncocpgfiaoituanbr`
- 열린국회정보 `nojepdqqaweusdfbi`

기능:
- 표결 API가 지원하는 재임 대수의 본회의 표결 의안 목록을 조회한다.
- 의원별 찬성, 반대, 기권, 불참 기록을 수집한다.
- 먼저 의원명 기준 일괄 조회를 시도하고, 부족하면 `BILL_ID`별 조회로 보완한다.
- 병렬 처리와 캐시를 사용해 반복 조회 시간을 줄인다.
- 일부 표결 조회가 실패해도 전체 워크플로우를 중단하지 않고 성공/실패 건수를 분리한다.

### 2.7 정당 다수 입장 일치도 분석 Tool

노드명: `analyze_party_alignment`

사용 API:
- 열린국회정보 `ncocpgfiaoituanbr`
- 열린국회정보 `nojepdqqaweusdfbi`

기능:
- `BILL_ID`별 정당 표결 분포를 계산한다.
- 공식 당론이 아니라 같은 정당 의원 다수가 선택한 표결값을 기준으로 정당 다수 입장을 계산한다.
- 의원 표결이 정당 다수 입장과 같으면 `일치`, 다르면 `이탈`, 불참이면 `불참`으로 분류한다.
- 무소속, 정당 내 표결자 1명, 찬성/반대/기권 동률 등은 `판정 제외`로 분리한다.

### 2.8 표결 통계 해석 Tool

노드명: `interpret_vote_statistics`

사용 데이터:
- 표결 요약
- 정당 다수 입장 일치도 요약
- 분석 대상 표결 날짜 범위
- 불참 및 추정 불참 정보

기능:
- 표결 통계를 해석할 때 사용자가 과잉 일반화하지 않도록 주의사항을 생성한다.
- 최근 N건 기준인지 전체 기준인지, 특정 날짜에 표결이 몰려 있는지, 불참 비중이 높은지 등을 설명한다.
- LLM 호출 실패 또는 quota 초과 시 규칙 기반 해석으로 대체한다.

### 2.9 최근 의원 관련 뉴스 검색 Tool

노드명: `search_recent_member_news`

사용 API:
- Google News RSS

기능:
- `{의원이름} 국회의원 {정당명}` 쿼리로 최근 뉴스 10건을 검색한다.
- 검색 결과의 제목, 언론사, 발행일, URL, snippet을 수집한다.
- LLM을 사용해 최근 공개 이슈를 요약한다.
- 열린국회정보 API 기반 공식 의정활동 데이터와 웹 검색 기반 맥락을 구분한다.

### 2.10 의원 활동 핵심 브리핑 Tool

노드명: `generate_activity_briefing`

사용 데이터:
- 의원 프로필
- 발의법안 통계
- 공동발의 네트워크 요약
- 표결 요약
- 정당 일치도 요약
- 최근 뉴스 수

기능:
- 사용자가 차트와 표를 보기 전에 읽을 수 있는 핵심 브리핑을 생성한다.
- 제공된 JSON 지표만 근거로 사용하고, 정치적 성과나 의도를 단정하지 않도록 제한한다.

### 2.11 최종 요약 생성 Tool

노드명: `summarize_member_activity`

기능:
- 앞선 모든 노드의 결과를 모아 최종 Markdown 요약을 생성한다.
- 의원 프로필, 조회 범위, 발의법안 요약, 공동발의 네트워크, 표결 요약, 정당 일치도, 최근 이슈, 오류 메시지를 정리한다.

## 3. LangGraph 워크플로우 구조와 병렬 처리 조건

`src/member_activity_workflow.py`의 `build_member_activity_graph` 기준 현재 워크플로우는 단순 직렬 구조가 아니라 fan-out/fan-in 병렬 구조이다.

### 3.1 전체 실행 구조

```text
START
-> normalize_user_request
-> get_member_info

get_member_info 이후 병렬 fan-out:
├─ search_member_bills
│  ├─ analyze_legislative_interests
│  └─ analyze_cosponsor_network
├─ get_all_member_votes
│  └─ analyze_party_alignment
│     └─ interpret_vote_statistics
└─ search_recent_member_news

fan-in:
analyze_legislative_interests
analyze_cosponsor_network
interpret_vote_statistics
search_recent_member_news
-> generate_activity_briefing
-> summarize_member_activity
-> END
```

### 3.2 병렬 처리 조건

병렬화 기준은 **서로의 산출물을 기다릴 필요가 없는 작업만 동시에 실행**하는 것이다. 현재 병렬 처리 조건은 다음과 같다.

| 구간 | 병렬 실행 노드 | 병렬 가능한 이유 |
|---|---|---|
| 의원 기본정보 조회 이후 | `search_member_bills`, `get_all_member_votes`, `search_recent_member_news` | 세 노드는 모두 의원명, 정당, 재임 대수 등 기본정보만 필요하며 서로의 결과를 필요로 하지 않는다. |
| 발의법안 조회 이후 | `analyze_legislative_interests`, `analyze_cosponsor_network` | 두 노드는 모두 발의법안 조회 결과를 입력으로 사용하지만, 서로의 분석 결과는 필요로 하지 않는다. |
| 표결 상세 조회 이후 | `analyze_party_alignment` → `interpret_vote_statistics` | 정당 일치도는 표결 상세 조회 결과가 필요하므로 병렬화하지 않고 순차 실행한다. 표결 해석 메모도 정당 일치도 결과를 함께 사용하므로 그 이후 실행한다. |
| 최종 종합 | `generate_activity_briefing` | 입법 관심 분야, 공동발의 네트워크, 표결 해석, 최근 뉴스 결과가 모두 모인 뒤 실행되는 fan-in 노드이다. |

### 3.3 상태 공유와 병합

각 노드는 `MemberActivityState`를 공유한다. 병렬 노드들이 동시에 같은 상태 키를 갱신할 수 있는 경우에는 충돌을 막기 위해 reducer를 적용한다.

- `errors`: 여러 병렬 노드가 오류 메시지를 추가할 수 있으므로 `merge_unique_messages`로 중복 없이 병합한다.
- `llm_quota_exhausted`: 여러 LLM 노드 중 하나라도 quota 초과를 감지하면 전체 상태에 반영되도록 `bool_or`로 병합한다.

이 구조 덕분에 발의법안 조회, 표결 조회, 최근 뉴스 검색처럼 독립적인 작업은 동시에 진행하면서도, 최종 브리핑은 필요한 중간 결과가 모두 준비된 뒤 안정적으로 생성된다.

## 4. 실제 작성한 System Prompt

아래 system prompt들은 모두 `src/member_activity_workflow.py`에 실제로 작성된 내용이다. 각 prompt는 `ChatGoogleGenerativeAI` 호출 시 `SystemMessage(content=system_prompt)`로 전달된다.

### 4.1 입법 관심 분야 분석 System Prompt

사용 함수:
- `generate_legislative_interest_analysis`

```text
당신은 국회의원 활동 데이터를 분석하는 정책 리서치 어시스턴트입니다. 정치적 호불호나 성과 평가를 하지 말고, 제공된 발의법안 통계와 법안 목록만 근거로 입법 관심 분야를 해석하세요. 추정은 반드시 '가능성', '신호', '해석상 주의'처럼 조심스럽게 표현하세요.
```

### 4.2 공동발의 네트워크 분석 System Prompt

사용 함수:
- `generate_cosponsor_network_analysis`

```text
당신은 국회의원 공동발의 네트워크를 해석하는 입법 데이터 분석가입니다. 제공된 통계만 근거로 협업 패턴을 설명하고, 정치적 친소관계나 실제 정책 동의를 단정하지 마세요.
```

### 4.3 표결 통계 해석 System Prompt

사용 함수:
- `generate_vote_interpretation_analysis`

```text
당신은 국회의원 표결 통계를 해석할 때 사용자의 과잉 일반화를 막는 데이터 리서치 어시스턴트입니다. 제공된 JSON 통계만 근거로 해석 주의사항을 작성하고, 통계에 없는 정치적 의도나 사실을 추정하지 마세요.
```

### 4.4 의원 활동 핵심 브리핑 System Prompt

사용 함수:
- `generate_activity_briefing`

```text
당신은 국회의원 활동 데이터의 첫 화면 브리핑을 작성하는 데이터 리서치 어시스턴트입니다. 제공된 JSON 지표만 근거로 사용하고, 정치적 성과·의도·호불호를 단정하지 마세요.
```

### 4.5 최근 의원 관련 뉴스 요약 System Prompt

사용 함수:
- `generate_recent_news_analysis`

```text
당신은 국회의원 관련 최근 공개 이슈를 요약하는 리서치 어시스턴트입니다. 제공된 뉴스 검색 결과만 근거로 요약하고, 사실관계나 의도를 단정하지 마세요. 공식 의정활동 데이터가 아니라 웹 검색 기반 맥락이라는 점을 명확히 구분하세요.
```

## 5. System Prompt 설계 기준

이 프로젝트의 system prompt는 공통적으로 다음 원칙을 따른다.

- 제공된 JSON 통계와 API 조회 결과만 근거로 분석한다.
- 정치적 호불호, 성과 평가, 의도 추정은 하지 않는다.
- 공식 데이터와 웹 검색 기반 데이터를 구분한다.
- 추정이 필요한 경우 단정하지 않고 조심스럽게 표현한다.
- LLM 호출이 실패하면 규칙 기반 fallback으로 대체한다.

## 6. 사용한 외부 데이터 소스

열린국회정보 API:

| API 서비스 | API 코드 | 사용 목적 |
|---|---:|---|
| 국회의원 정보 통합 API | `ALLNAMEMBER` | 의원 기본정보, 정당, 선거구, 위원회, 재임 대수 |
| 국회의원 발의법률안 | `nzmimeepazxkubdpn` | 발의법안 목록, 대표발의/공동발의, 처리결과, 소관위원회 |
| 의안별 표결현황 | `ncocpgfiaoituanbr` | 본회의 표결 의안 목록, `BILL_ID`, 표결일 |
| 국회의원 본회의 표결정보 | `nojepdqqaweusdfbi` | 의원별 표결 기록, 정당별 표결 분포 계산 |

기타 외부 소스:

| 소스 | 사용 목적 |
|---|---|
| Google News RSS | 최근 의원 관련 뉴스 검색 |
| Gemini `gemini-2.5-flash-lite` | 발의법안, 공동발의 네트워크, 표결 통계, 최근 뉴스, 핵심 브리핑 요약 |
