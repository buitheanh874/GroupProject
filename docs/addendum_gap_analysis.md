# Gap Analysis: Addendum vs. Standard Paper Structure
**Target Document**: `SUPPLEMENT_Addendum_Workflows_Settings.md` (Matches *Revised Addendum.docx*)
**Standard**: Top-tier RL Traffic Signal Control Papers (PressLight, CoLight).

This analysis highlights structural and content gaps that need to be addressed to align the Addendum with academic standards.

## 1. Critical Gaps (Must-Have)

| Component | Status in Addendum | Standard Requirement | Action Item |
| :--- | :--- | :--- | :--- |
| **Abstract** | **Missing**. Only has "1. Overview". | concise summary of Motivation, Method, and Key Results. | **Add "Abstract" section** before Section 1. |
| **Introduction** | **Weak**. "Administrative" style ("This document provides..."). | **Scientific** style: Problem (Congestion) -> Challenges -> Proposed Solution (MARL+SMDP). | **Rewrite Section 1** to focus on the scientific problem and contribution. |
| **Related Work** | **Missing**. | Comparison with Fixed-Time, Max-Pressure, and standard RL to position the work. | **Add "Section 2: Related Work"**. Brief review of PressLight/CoLight. |
| **Why Global State?** | **Implicit** in Section 2.2.1. | Explicit theoretical justification (Dec-POMDP -> MDP). | **Enhance Section 2.2.1** with "Design Rationale" citing CoLight (observability). |
| **Why SMDP?** | **Implicit** in Section 2.2.3. | Explicit justification using Max Pressure theory. | **Enhance Section 2.2.3** connecting Reward to Max Pressure optimization. |
| **Conclusion** | **Missing**. | Summary of contributions and future work. | **Add "Conclusion" section** at the end. |

## 2. Refinement Details

### 2.1 Renaming & Restructuring
*   Change **"1. ADDENDUM OVERVIEW"** to **"1. INTRODUCTION"**.
*   Insert **"2. RELATED WORK"** after Introduction.
*   Shift existing "2. PROBLEM FORMULATION" to **"3. PROBLEM FORMULATION"**.
*   Shift existing "3. BASELINES..." to **"4. EXPERIMENTAL SETUP"**.
*   Shift existing "6. RESULTS..." to **"7. RESULTS & ANALYSIS"**.

### 2.2 Content Enhancements
*   **Methodology (Section 3)**:
    *   Add a visual interaction diagram (Mermaid) showing the Agent-Environment loop with Global Broadcast.
    *   Cite "PressLight" when explaining the "Queue Length" features.
*   **Experimental Setup (Section 4)**:
    *   Explicitly state the "Research Questions" (RQs) we aim to answer (e.g., RQ1: Does SMDP improve over Fixed-time? RQ2: Does Global State help cooperation?).
*   **Results (Section 7)**:
    *   Replace placeholders with "Expected trends" based on initial findings (RL > Fixed > Max-Pressure).

## 3. Checklist for User Approval

1.  [ ] **Confirm Target File**: Is `SUPPLEMENT_Addendum_Workflows_Settings.md` the correct source for the .docx?
2.  [ ] **Approve Restructure**: Do you agree with renaming sections to (Intro -> Related -> Method -> Exp -> Results)?
3.  [ ] **Approve Narrative**: Do you want the text to sound more "Academic" (like a paper) or keep it "Technical" (like a manual)? -> *Recommendation: Academic.*
