# Standardized Chain-of-Thought Format for Virtual Expert Routing

## Overview

This document defines the standardized CoT (Chain-of-Thought) format used to train models for virtual expert routing. The format provides a consistent way to express reasoning across all expert types.

## Format Specification

```
<expert>: <spec> -> <result>
```

### Components

| Component | Description | Examples |
|-----------|-------------|----------|
| `<expert>` | The expert token that triggers routing | `multiply`, `time`, `schedule`, `chat` |
| `<spec>` | Specification in expert's format | `256 * 4`, `Asia/Tokyo`, `eggs:16 \| -3 -4 *2` |
| `<result>` | Output from expert execution | `1024`, `14:30 JST`, `18` |

### Separators

- `:` separates expert from spec
- `->` separates spec from result
- `|` separates entities from operations (in word problems, CSP)

## Expert Types

### 1. Simple Arithmetic

**Expert tokens:** `multiply`, `add`, `subtract`, `divide`

**Spec format:** Mathematical expression

```
multiply: 256 * 4 -> 1024
add: 100 + 200 -> 300
subtract: 50 - 17 -> 33
divide: 144 / 12 -> 12
```

### 2. Word Problems (GSM8K-style)

**Expert token:** `word_problem`

**Spec format:** `entity:initial_value | operations`

Operations use prefix notation:
- `+N` add N
- `-N` subtract N
- `*N` multiply by N
- `/N` divide by N

```
word_problem: eggs:16 | -3 -4 *2 -> 18
word_problem: marbles:10 | -3 +5 -> 12
word_problem: price:40 | *0.75 -> 30
```

**Interpretation:**
- `eggs:16 | -3 -4 *2` = Start with 16, subtract 3, subtract 4, multiply by 2
- `16 - 3 = 13`, `13 - 4 = 9`, `9 * 2 = 18`

### 3. CSP Scheduling

**Expert token:** `schedule`

**Spec format:** `tasks | constraints`

Tasks: `Name:duration,Name:duration,...`
Constraints: `constraint1,constraint2,...`

```
schedule: Alice:2hr,Bob:1hr | no_overlap -> Alice:9-11,Bob:11-12
schedule: gym:1hr,lunch:1.5hr,meeting:2hr | sequential -> gym:9-10,lunch:10-12,meeting:12-14
```

**Supported constraints:**
- `no_overlap` - Tasks cannot overlap
- `sequential` - Tasks must run in order
- `before(A,B)` - A must finish before B starts

### 4. Time Queries

**Expert token:** `time`

**Spec format:** IANA timezone or location alias

```
time: Asia/Tokyo -> 14:30 JST
time: Europe/London -> 09:15 GMT
time: America/New_York -> 04:30 EST
```

**Location aliases:**
- `Tokyo` -> `Asia/Tokyo`
- `London` -> `Europe/London`
- `New York` -> `America/New_York`
- `Paris` -> `Europe/Paris`
- `Sydney` -> `Australia/Sydney`

### 5. Passthrough (Chat)

**Expert token:** `chat`

**Spec format:** Empty (no spec needed)

```
chat: -> [response]
```

Used when no virtual expert should handle the query.

## Query to CoT Mapping

### Extra Instructions

The model must learn to ignore extra instructions and still extract the core task:

| Query | CoT |
|-------|-----|
| `What is 256 * 4?` | `multiply: 256 * 4 -> 1024` |
| `What is 256 * 4? Answer briefly.` | `multiply: 256 * 4 -> 1024` |
| `Calculate 256 * 4, show work` | `multiply: 256 * 4 -> 1024` |
| `Current time in Tokyo` | `time: Asia/Tokyo -> 14:30 JST` |
| `Time in Tokyo, answer in french` | `time: Asia/Tokyo -> 14:30 JST` |

### Word Problem Extraction

Complex word problems are reduced to entity tracking:

**Query:** "Janet has 16 eggs. She eats 3 and bakes 4. She sells the rest for $2 each. How much?"

**Analysis:**
1. Initial: 16 eggs
2. Operation: -3 (eats)
3. Operation: -4 (bakes)
4. Operation: *2 (sells for $2 each)

**CoT:** `word_problem: eggs:16 | -3 -4 *2 -> 18`

## Training Flow

```
User Query
    ↓
Model generates CoT: "<expert>: <spec> -> ..."
    ↓
Hidden states at "<expert>:" token
    ↓
MoE router detects expert activation
    ↓
Virtual expert parses <spec>
    ↓
Expert executes and produces result
```

## Verified Rewards

The reward function verifies CoT by actually executing:

| Condition | Reward |
|-----------|--------|
| Correct expert + correct result | 1.0 |
| Correct expert + wrong result | 0.7 |
| Correct expert + parse fail | 0.5 |
| Wrong expert | 0.3 |
| Format failure | 0.0 |

## Extensibility

To add a new expert type:

1. Define the expert token (e.g., `code_exec`)
2. Define the spec format (e.g., `language:code`)
3. Implement execution function
4. Add training examples

Example for code execution:
```
code_exec: python:print(2+2) -> 4
code_exec: javascript:console.log("hello") -> hello
```

## References

- cot_vocab_alignment experiment: Shows CoT creates vocabulary-aligned classifiers
- csp_cot_gsm8k: Word problem extraction with trace verification
- csp_virtual_expert: CSP detection and solving
