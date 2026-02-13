# NL Spec Format Guide

Use this guide as a structural template when generating natural-language specifications in the Attractor spec style. Every rule below is derived from three canonical specs: `attractor-spec.md`, `coding-agent-loop-spec.md`, and `unified-llm-spec.md`.

---

## 1. Document Skeleton

Every spec follows this exact top-level structure:

```markdown
# {Spec Title}

{1-3 sentence summary paragraph. States WHAT the spec defines, WHO it's for,
and HOW it relates to companion specs. Written as a declarative fact, not a sales pitch.}

---

## Table of Contents

1. [Overview and Goals](#1-overview-and-goals)
2. [{Core Domain Section}](#2-...)
3. [{Core Domain Section}](#3-...)
...
N-1. [{Last Feature Section}](#...)
N. [Definition of Done](#n-definition-of-done)

---

## 1. Overview and Goals
### 1.1 Problem Statement
### 1.2 Why {Key Design Choice}
### 1.3 Design Principles
### 1.4 Architecture (or Layering)
### 1.5 Reference Projects (optional)
### 1.6 Relationship to {Companion Spec} (if applicable)

---

## 2. {First Core Domain}
### 2.1 ...
...

---

## N. Definition of Done
### N.1 {Feature Area A}
### N.2 {Feature Area B}
...
### N.K Cross-Feature Parity Matrix
### N.K+1 Integration Smoke Test

---

## Appendix A: {Reference Tables}
## Appendix B: {Format References}
## Appendix C: Design Decision Rationale (optional)
```

**Rules:**
- H1 (`#`) appears exactly once: the spec title.
- Top-level sections use `## N. Title` with integer numbering starting at 1.
- Subsections use `### N.M Title` with decimal numbering.
- Sub-subsections use `#### Descriptive Title` (no triple numbering like `1.1.1`).
- Horizontal rules (`---`) separate every top-level section.
- The Table of Contents uses a numbered markdown list with anchor links.
- "Definition of Done" is always the LAST numbered section, before appendices.
- Appendices use letter prefixes: `Appendix A`, `Appendix B`, etc.

---

## 2. Section 1: Overview and Goals

This section always contains these subsections in order:

### 1.1 Problem Statement
- 1-2 paragraphs.
- States the problem concretely. Names specific pain points.
- Ends with a one-sentence declaration of what this spec solves.

### 1.2 Why {Key Design Choice}
- Justifies the single most important architectural bet (e.g., "Why DOT Syntax", "Why a Library Not a CLI").
- Uses a bulleted list with **bold lead-ins** explaining each reason.

### 1.3 Design Principles
- 4-6 principles, each as a **bold keyword phrase** followed by a paragraph.
- Pattern: `**Keyword phrase.** Explanation paragraph.`
- Example: `**Declarative pipelines.** The .dot file declares what the workflow looks like...`

### 1.4 Architecture / Layering
- ASCII box diagram in a fenced code block showing layers/components.
- Or a text flow: `PARSE -> VALIDATE -> INITIALIZE -> EXECUTE -> FINALIZE`
- Followed by a paragraph explaining how this spec relates to companion specs (what it owns vs. what it delegates).

### 1.5 Reference Projects (optional)
- Bulleted list of open-source projects.
- Each: `**name** (URL) -- Language. One sentence about what it demonstrates.`
- Explicitly state: "They are not dependencies; implementors may take inspiration."

### 1.6 Relationship to Companion Specs
- State which types/interfaces are imported from companion specs.
- Use file-relative links: `[Unified LLM Client](./unified-llm-spec.md)`.

---

## 3. Core Domain Sections (Section 2 through N-1)

Each feature section follows this internal pattern:

```
### N.1 Overview / Concept
    Plain English paragraph(s). What is it, why does it exist.

### N.2 Data Model
    RECORD / INTERFACE / ENUM definitions in pseudocode.

### N.3 Algorithm / Behavior
    FUNCTION pseudocode. This is the heart of each section.

### N.4 Configuration / Attributes
    Table: Key | Type | Default | Description

### N.5 Examples
    Concrete code/config in fenced code blocks.

### N.6 Variants / Policies / Mappings
    Tables listing modes, policies, or provider-specific mappings.
```

Not every subsection is required. The pattern is: **concept -> data structures -> algorithms -> configuration -> examples -> edge cases**.

---

## 4. Pseudocode Style

All pseudocode uses a language-neutral notation inside fenced code blocks (no language tag):

### Keywords (ALL CAPS)
```
FUNCTION, RETURN, IF, ELSE IF, ELSE, FOR EACH, WHILE, LOOP, END LOOP,
BREAK, CONTINUE, TRY, CATCH, RAISE, AND, OR, NOT, IN, IS, NONE, true, false
```

### Data Definitions
```
RECORD Name:           -- A data structure / struct / class
    field : Type       -- field with type annotation
    field : Type = val -- field with default value

INTERFACE Name:        -- An abstract interface / protocol / trait
    FUNCTION method(param: Type) -> ReturnType

ENUM Name:             -- An enumeration
    VALUE1    -- description
    VALUE2    -- description
```

### Functions
```
FUNCTION name(param1: Type, param2: Type) -> ReturnType:
    -- Comment uses double-dash prefix
    variable = expression
    IF condition:
        do_something()
    FOR EACH item IN collection:
        process(item)
    RETURN result
```

### Conventions
- Comments use `--` prefix (not `//` or `#`).
- Type annotations use `: Type` syntax.
- Optional/nullable types use `Type | None`.
- Union types use `Type | OtherType`.
- Generic collections use `List<T>`, `Map<K, V>`.
- No braces for blocks; indentation indicates scope.
- String concatenation uses `+`.
- Field access uses `.` notation.

---

## 5. Tables

Tables are used heavily. There are four canonical table formats:

### Attribute Reference Table
```markdown
| Key              | Type     | Default   | Description |
|------------------|----------|-----------|-------------|
| `attribute_name` | String   | `""`      | What it does. |
| `other_attr`     | Integer  | `0`       | What it controls. |
```

### Mapping / Translation Table
```markdown
| SDK Format       | OpenAI            | Anthropic          | Gemini            |
|------------------|-------------------|--------------------|-------------------|
| `unified_field`  | `provider_field`  | `provider_field`   | `provider_field`  |
```

### Policy / Mode Table
```markdown
| Mode             | Behavior |
|------------------|----------|
| `mode_name`      | What it does. |
```

### Parity Matrix (in Definition of Done)
```markdown
| Test Case                              | OpenAI | Anthropic | Gemini |
|----------------------------------------|--------|-----------|--------|
| Simple text generation                 | [ ]    | [ ]       | [ ]    |
| Tool calling works                     | [ ]    | [ ]       | [ ]    |
```

**Table rules:**
- Attribute names, enum values, and config keys are wrapped in backticks in table cells.
- Type column uses plain text: `String`, `Integer`, `Boolean`, `Duration`, `Float`.
- Default column wraps literal values in backticks: `` `""` ``, `` `0` ``, `` `false` ``.
- Description column is a single concise sentence.

---

## 6. Requirement Language

These specs do NOT use formal RFC-2119 (`MUST`/`SHALL`/`SHOULD`/`MAY`) pervasively. Instead they use three tiers:

### Primary style: Declarative facts
State what the system does as direct assertions:
- "The engine selects the next edge from the node's outgoing edges."
- "Each parallel branch receives an isolated clone of the context."

### For hard constraints: Bold or ALL CAPS imperative
- "Handlers **MUST** be stateless or protect shared mutable state."
- "Character-based truncation ... **MUST** always run first."
- "The adapter **MUST** use the provider's native API."

### For recommendations: "should" in lowercase
- "The adapter should accept them via `provider_options`."
- "Implementations should default to the latest available models."

### For contracts: Bulleted list with bold label
```markdown
**Handler contract:**
- Handlers MUST be stateless or protect shared mutable state.
- Handler panics MUST be caught by the engine and converted to FAIL outcomes.
- Handlers SHOULD NOT embed provider-specific logic.
```

---

## 7. Cross-References

### Between sections in the same spec
Use parenthetical references:
- `(Section 4.5)`
- `(see Section 8 for details)`
- `(detailed in Section 5)`

### Between companion specs
Use markdown file-relative links on first mention:
- `[Unified LLM Client Specification](./unified-llm-spec.md)`

After first mention, use short inline references:
- `the companion Unified LLM Client spec`
- `the Coding Agent Loop spec (Section 2.5)`

---

## 8. Definition of Done Section

This is the final numbered section. It defines acceptance criteria as checkboxes.

### Structure
```markdown
## N. Definition of Done

This section defines how to validate that an implementation of this spec
is complete and correct. An implementation is done when every item is
checked off.

### N.1 {Feature Area}

- [ ] Imperative statement of what must work
- [ ] Another concrete, testable requirement
- [ ] Third requirement

### N.2 {Another Feature Area}

- [ ] ...
```

### Rules
- Opening paragraph is always the same boilerplate (see above).
- Subsections mirror the spec's core sections (one DoD subsection per feature section).
- Each checkbox item is a single imperative sentence.
- Items are testable: they describe observable behavior, not vague qualities.
- Good: `- [ ] Edge selection follows the 5-step priority: condition match -> preferred label -> suggested IDs -> weight -> lexical`
- Bad: `- [ ] Edge selection works correctly`

### Cross-Feature Parity Matrix
A table where rows are test cases and columns are implementations/providers. Every cell is `[ ]`.

### Integration Smoke Test
The final subsection. Contains complete pseudocode exercising the system end-to-end with `ASSERT` statements.

```
-- 1. Parse
graph = parse_dot(DOT)
ASSERT graph.goal == "Create a hello world Python script"

-- 2. Validate
lint_results = validate(graph)
ASSERT no error-severity results in lint_results

-- 3. Execute with callback
outcome = run_pipeline(graph, context, callback = real_callback)

-- 4. Verify
ASSERT outcome.status == "success"
```

---

## 9. Recurring Structural Patterns

### The "Interface + Implementations" pattern
1. Define the interface with `INTERFACE Name:` pseudocode.
2. List built-in implementations, each as a subsection with its own pseudocode.
3. End with a "Custom {X}" subsection explaining extension.

Example from the specs:
- Interviewer Interface -> AutoApprove, Console, Callback, Queue, Recording implementations -> custom interviewer guidance.
- ProviderAdapter interface -> OpenAI, Anthropic, Gemini adapters -> adding a new provider.

### The "Layering and Ownership" pattern
When a spec depends on companion specs, explicitly state:
- What THIS spec owns (e.g., "orchestration, graph traversal, state management").
- What it delegates (e.g., "LLM communication is handled by the Unified LLM Client").
- That the pipeline definition does NOT change regardless of backend choice.

### The "Out of Scope" section (optional)
A dedicated section listing features that are intentionally excluded:
- Each item: **Bold feature name.** 1-2 sentences on what it would do, where the natural extension point is, and why it's excluded from THIS spec.

### The "Design Decision Rationale" appendix (optional)
An appendix with bold question-answer pairs:
- **Why {choice} instead of {alternative}?** Explanation paragraph.

### BNF-Style Grammars
For any custom DSL or expression language, provide a formal grammar in a fenced code block:
```
Expr     ::= Clause ( '&&' Clause )*
Clause   ::= Key Operator Literal
Key      ::= 'outcome' | 'context.' Path
Operator ::= '=' | '!='
```

### Example blocks
Concrete examples appear AFTER the formal definition, labeled with bold descriptive text:
```markdown
**Simple linear workflow:**

    digraph Simple { ... }

**Branching workflow with conditions:**

    digraph Branch { ... }
```

---

## 10. Formatting Conventions

- **Bold** for key terms on first introduction, design principle names, and feature names.
- `Backtick` for attribute names, config keys, enum values, type names, file names, and tool names.
- *Italic* is used rarely (almost never in these specs).
- Fenced code blocks for ALL pseudocode, grammars, examples, and file structures.
- Bulleted lists for enumerating items without ordering significance.
- Numbered lists ONLY for ordered sequences (steps in an algorithm described in prose).
- Em dashes (`--`) used in flowing prose for parenthetical asides.
- No emoji anywhere in the spec.
- Sentence fragments are acceptable in table description columns.
- File paths and directory structures use fenced code blocks.

---

## Quick Reference: Generating a New Spec

When generating a spec from scratch, follow these steps:

1. **Write the H1 title and summary paragraph.**
2. **Draft the Table of Contents** with section names (refine later).
3. **Write Section 1 Overview and Goals** with Problem Statement, Why {X}, Design Principles, Architecture.
4. **For each core feature section:**
   - Start with a concept paragraph.
   - Define data structures (`RECORD`, `INTERFACE`, `ENUM`).
   - Write the algorithm as pseudocode `FUNCTION` blocks.
   - Add attribute/config reference tables.
   - Include concrete examples.
   - Document variants, policies, and edge cases.
5. **Write the Definition of Done** with checkboxes mirroring each feature section, a parity matrix, and an integration smoke test.
6. **Add Appendices** for complete reference tables, format specs, and design rationale.
7. **Go back and add cross-references** (`Section N.M`) everywhere a concept is defined elsewhere.
8. **Finalize the Table of Contents** to match actual sections.
