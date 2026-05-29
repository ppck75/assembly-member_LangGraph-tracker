# 🕵️‍♂️ assembly-member_LangGraph-tracker

> **LangGraph 기반 대한민국 국회의원 활동 및 입법 프로세스 추적 AI 에이전트**
>
> 본 프로젝트는 디지털 공론장(Digital Public Sphere) 및 감시 저널리즘(Watchdog Journalism) 이론을 고도화된 AI 파이프라인으로 구현한 시스템입니다. 복잡하고 파편화된 국회 데이터를 시민들이 쉽게 이해하고 추적할 수 있도록 돕는 것을 목표로 합니다.    
URL: https://national-assembly.streamlit.app/  
---

## 🌟 주요 기능 (Key Features)


---

## 🏗️ 시스템 아키텍처 및 워크플로우 (Architecture)

본 시스템은 데이터 수집/전처리 레이어와 LangGraph 기반의 추론/에이전트 레이어로 분리되어 상호작용합니다.

```text
[열린국회정보 Open API]
         │
         ▼ (실시간 수집 및 분류)
┌────────────────────────────────────────┐
│                                          │ ──► 
│   
└────────────────────────────────────────┘
         │
         ▼ )
┌────────────────────────────────────────┐
│     의원_활동_추적_랭그래프_워크플로우.ipynb  │ ──► LangGraph State Graph 기반 
│     (LangGraph Agent Workflow Layer)   │     LLM 맥락 요약 및 답변 생성
└────────────────────────────────────────┘
```

---

## 📂 프로젝트 구조 (Project Structure)

📂
├── 의원_활동_추적_랭그래프_워크플로우. 
├── 
├── 
├── 
├── 
└── .

---

## 🛠️ 기술 스택 (Tech Stack)
- Language: Python 3.10+  
- Frameworks: LangGraph, LangChain
- Data Source: 대한민국 국회 열린데이터광장(Open API) 및 실시간 Google News 수집
- Streamlit: 수집된 의정 활동 데이터 및 LLM 추론 결과를 직관적인 웹 인터페이스로 시각화
---
