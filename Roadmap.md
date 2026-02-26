# Project Documentation: Self-Speculating Analog In-Memory Computing

## 1. Background & Architectural Evolution

This section details the transition from traditional digital-mimicking analog architectures to the proposed residual physics-based architecture.

### 1.1 Limitation of Conventional Architectures: Bit-Slicing

The standard industry approach to high-precision Analog Compute-in-Memory (ACIM) forces analog devices to mimic digital binary logic.

* **The Mechanism:** A weight (e.g., 8-bit integer) is sliced into multiple low-precision segments (e.g., four 2-bit slices). Each slice represents a strictly defined power of 2 ($2^0, 2^2, 2^4, 2^6$).
* **The Hardware Bottleneck:**
  * **Redundant Digitization:** Each slice requires its own ADC operation (or multiple cycles of a shared ADC).
  * **Digital Reconstruction Overhead:** The results must be digitally shifted and added. This consumes significant dynamic power and chip area for shift-adders.
  * **MSB Sensitivity:** The Most Significant Bit (MSB) arrays are unforgiving. A slight analog variation in the $2^6$ array translates to a massive error in the final output, requiring expensive error-correction codes (ECC) or extreme device endurance.

### 1.2 Proposed Base Architecture: Residual "True Analog"

We adopt the architecture proposed by *Song et al. (Science, 2024)*, which treats quantization noise and device write noise as a unified physical "Residual."

* **The Concept:** Instead of pre-slicing bits, we write values sequentially based on physical feedback.
    1. **Array 1 (Coarse)** attempts to write the target value $W$. Due to variability, it settles at $W_{actual}$.
    2. The system measures the error $\epsilon_1 = W - W_{actual}$.
    3. **Array 2 (Residual)** is programmed to store $\epsilon_1$, scaled by a gain factor $K$ to maximize the dynamic range of the device.
    4. This repeats for Arrays 3 and 4.
* **The Analog Advantage:** The summation of stored values occurs via **Kirchhoff’s Current Law (KCL)** on the bitline *before* the ADC involves.
    $$ I_{total} = I_{Array1} + (I_{Array2}/K) + (I_{Array3}/K^2) $$
* **Key Implication:** High precision ($10^{-15}$) is achieved with a single readout interface, removing the digital reconstruction overhead.

![Residual Architecture](./Idea%20Slide%20Figure/main.png)

---

## 2. Core Innovation: Zero-Overhead Speculative Decoding

Large Language Models (LLMs) do not require scientific-grade precision for every token. We leverage the physical properties of the Residual Architecture to implement **Speculative Decoding** natively in hardware, without the extra memory cost required by software speculative decoding (which needs a separate "Draft Model").

### 2.1 The "Hardware-Native" Draft Model

In the Residual Architecture, **Array 1** (the uncompensated array) is a coarse analog projection of the full residual-programmed model. Its lower effective precision comes from reading only the first residual stage (plus analog/device/readout noise), not from bit-plane weight quantization.

* **Magnitude Dominance:** Array 1 typically captures >95% of the signal energy (the "shape" of the weight vector). Arrays 2-4 only provide the "fine tuning."
* **The Strategy:** We repurpose Array 1 as a standalone "Draft Model." By reading only Array 1, we obtain a fast, low-energy approximation of the next token.

### 2.2 Dual-Mode Inference Strategy

Software controls the hardware state to toggle between "High Throughput" and "High Precision."

#### Mode A: Draft Mode (High Speed)

* **Hardware State:** Draft-mode compute uses the configurable precision policy (§4.1). By default, Switch 1 is Closed (Active) and Switches 2-4 are Power Gated, but selected blocks/layers can enable additional arrays to run in full precision even during Draft Mode.
* **ADC Config:** ADC-Draft is always active (low-resolution, e.g., 4-bit). For blocks/layers configured to use **full precision** in Draft Mode, we activate Arrays 1-4 and use both ADCs, then digitally combine and store the full-precision result for possible reuse in Verify Mode.
* **Precision Interpretation:** In Draft Mode, "low precision" means coarse analog readout (Array 1 + low-resolution ADC path), not quantized weights.
* **Latency:** Ultra-low (e.g., 5ns).
* **Usage:** Used to auto-regressively generate a sequence of $K$ future tokens (e.g., 5 tokens ahead).

#### Mode B: Verify Mode (High Precision)

* **Hardware State:** Verification uses the residual path described in §3.1. Since the verifier input stream can differ from the draft input stream after a mismatch, the estimator must support a configurable policy for whether to reuse stored draft results or to re-read:
  * **Reuse enabled:** for drafted tokens, reuse stored draft results when possible:
    * If the block was executed in default Draft Mode (Array 1 only): reuse the stored base value and read Arrays 2-4 via ADC-Residual to obtain the correction.
    * If the block was executed in full-precision Draft Mode (Arrays 1-4 + both ADCs): reuse the stored full-precision output directly (no additional analog read for that block).
  * **Reuse disabled:** for drafted tokens, re-read Array 1 (and Arrays 2-4) by activating all arrays and using both ADCs, then digitally combine.
  * For the verifier’s bonus token (no stored draft): activate Arrays 1-4 and use both ADCs, then digitally combine.
* **ADC Config:** ADC-Residual is high-resolution (e.g., 12-bit SAR/Pipe, multi-sample averaging if needed for effective precision).
* **Latency:** High (e.g., 50ns).
* **Usage:** Used to validate the sequence generated by the Draft Mode.

### 2.3 Spatial Pipelined Verification

Why is this faster? Conventional analog inference is limited by **Setup Latency** (charging long capacitive bitlines).

* **Conventional (Stop-and-Go):** Setup -> Read Token 1 -> Setup -> Read Token 2. The setup penalty is paid $N$ times.
* **Speculative Burst:**
    1. We generate $K$ tokens quickly using Draft Mode (e.g., $K=5$).
    2. We feed the drafted tokens into the High-Precision Verifier in a **Burst**, and compute $K+1$ pipelined verify steps (including the bonus token).
    3. The bitlines remain charged/active. The ADC operates in "Throughput Mode" rather than "Latency Mode."
    4. The Setup penalty is paid only **once** for the entire burst.

---

## 3. Hardware Architecture Implementation

To maximize the efficiency of the Dual-Mode strategy, we redesign the peripheral readout circuitry (ADC Layout). The standard single-ADC design creates a data bottleneck; our proposed design enables reuse and parallelism.

### 3.1 Design Choice 1: The "1+3 Split" ADC Layout

We replace the single monolithic ADC with a decoupled, heterogeneous readout system.

* **Component A: ADC-Draft (4-bit Flash/SAR)**
  * Connected permanently to **Array 1**.
  * Speed: Very Fast (<1 cycle).
  * Power: Negligible.
* **Component B: ADC-Residual (12-bit Pipeline/SAR)**
  * Connected via switches to the sum of **Arrays 2, 3, and 4**.
  * Dynamic Range: Tuned for small currents (high gain).
* **The "Non-Destructive Reuse" Logic:**
  * During Draft Mode (Default): Only **ADC-Draft** fires (Array 1). Result is stored in a register ($D_{reg}$).
  * During Draft Mode (Full Precision Blocks): Activate Arrays 1-4 and use both ADCs, then digitally combine and store the full-precision result (buffered for possible reuse in Verify Mode).
  * During Verify Mode (Reuse enabled, Array-1-only draft): We **do not re-read Array 1**. We assume $D_{reg}$ is the base value. We activate Arrays 2-4, read them via **ADC-Residual** to get correction $C$, and digitally compute $Final = D_{reg} + C$.
  * During Verify Mode (Reuse enabled, full-precision draft): We reuse the stored full-precision output directly (no additional analog read for that block).
  * For the verifier’s **bonus token** (where there is no stored $D_{reg}$ from Draft Mode), we perform a full analog read by activating **all 4 arrays** and using **both ADCs**, then digitally combine the two ADC outputs to form the full-precision result.
  * **Configurable Reuse:** Since the verifier’s input stream may differ from the draft’s input stream after a mismatch, the estimator will support a knob to disable reuse. When reuse is disabled, verification re-reads Array 1 (and Arrays 2-4) for drafted tokens as well (activate Arrays 1-4, use both ADCs, then digitally combine).
  * **Energy Savings:** Removes the redundant active power of reading the heavy Array 1 during verification phases.

### 3.2 Design Choice 2: Predictive Delta Readout (Combinable with Opt 1)

LLM outputs exhibit high **Temporal Locality**. The voltage on a bitline for Token $N$ is statistically very close to Token $N-1$ (due to embedding similarity or attention stability).

* **Mechanism:**
    1. The digitized output of Token $N-1$ is stored in a Feedback Register.
    2. A **DAC (Digital-to-Analog Converter)** injects a negative current corresponding to Token $N-1$ onto the bitline during the read of Token $N$.
    3. **The Result:** The ADC bitline now holds $\Delta V = V_{token\_N} - V_{token\_N-1}$.
* **Gain:** The $\Delta V$ is much smaller than the absolute $V$. This signal compression allows us to lower the required dynamic range of **ADC-Residual**, achieving effective 16-bit precision using only 8-10 physical bits.

![Speculative Inference Timeline](./Idea%20Slide%20Figure/opt.png)

---

## 4. Algorithm-Hardware Co-Design (Optimizations)

Hardware alone cannot guarantee efficiency. If the "Draft Model" (Array 1) is too inaccurate, the speculation will fail, and we will revert to slow verification too often. We use two algorithmic techniques to maximize the **Draft Acceptance Rate**.

### 4.1 Optimization 1: Sensitivity-Aware Draft Allocation

Not all layers in a Transformer affect the output equally. Some layers act as "Memory Retrieval" (High Sensitivity), while others process robust features (Low Sensitivity).

* **Profiling:** We use Hessian-based sensitivity analysis (or magnitude pruning scores) to classify layers into **Robust** and **Sensitive**.
* **Dynamic Allocation Policy (Submodule Granularity):** Rather than a single precision per layer, we assign a Draft precision policy for three compute blocks in each layer:
  * **QKV projection (Q/K/V):** draft precision vs. full precision.
  * **$W_O$ projection:** draft precision vs. full precision.
  * **FFN (MLP):** draft precision vs. full precision (for Residual Analog CIM, “full” can mean enabling residual arrays beyond Array 1).
* **Control Flow:** A lightweight controller looks up the per-layer policy ("Precision ID") before triggering the Draft Burst, enabling higher precision only where it improves acceptance rate the most.

### 4.2 Optimization 2: Noise-Injection Adaptation (Fine-Tuning)

A standard INT8 quantization model assumes perfect integer math. When deployed on **Array 1**, the analog write noise (Gaussian perturbation) causes "Golden" weights to drift, destroying the accuracy of the draft.

* **The Mismatch:** Standard models have sharp minima. A small analog drift pushes the weight out of the optimal valley, causing token flips.
* **The Solution:** Noise-Injection Training (NIT).
  * **Procedure:** We fine-tune the LLM for a small number of steps (e.g., 1000 steps).
  * **Injection:** During the Forward Pass, we add random noise $\mathcal{N}(0, \sigma_{device})$ to the weights, where $\sigma_{device}$ matches the measured physical write variability of the memristor.
  * **Result:** The optimizer seeks "Flat Minima"—regions where weight variations do not change the output logits.
* **Metric Improvement:** Increases Draft Acceptance Rate from ~60% (unusable) to >85% (highly efficient), enabling the speculative pipeline to maintain throughput.

## 5. Evaluation Methodology

To validate the proposed **Self-Speculating Analog Architecture**, we employ a hybrid simulation approach. Since the specific hardware (the Dual-ADC Analog SoC) utilizes novel readout logic not yet available in silicon, we decouple the evaluation into two distinct modules: a **Functional Simulator** (for accuracy and acceptance rates) and a **Hardware Estimator** (for energy, latency, and area).

### 5.1 Simulator Framework Overview

The verification pipeline consists of two interacting components:

1. **The Accuracy Engine (PyTorch):** Based on the open-source **`gpt-fast`** (by Meta PyTorch Team). We will modify this to simulate the algorithmic behavior, injecting device/readout noise to mimic analog draft behavior and modeling finite ReRAM interface precision (DAC/ADC) where applicable.
2. **The Performance Calculator (`selfspec-calculator`, Python):** A parametric tool that takes usage statistics from the Accuracy Engine (e.g., "How many tokens were accepted?") and maps them to physical metrics using two configs (`model.yaml` and `hardware.yaml`).

### 5.2 Part A: Functional Simulation (Accuracy & Speculation)

We intend to modify the `gpt-fast` inference loop to emulate the physical behavior of the residual arrays.

* **Base Model:** **[Models around 1B, like Llama 3 1B, Qwen 2.5 1.5B or GPT-2-XL]**
* **Datasets:** **[PLACEHOLDER: e.g., WikiText-2, C4, or HumanEval]**

#### 5.2.1 Noise & Interface-Precision Modeling

To simulate **Array 1 (The Draft Model)** and residual verification behavior:

1. **No weight quantization in the ReRAM path:** keep the model weights represented by the residual analog decomposition (Array 1 + residual arrays), not digital bit-plane quantized weights.
2. **Draft analog approximation:** model draft behavior by reading only Array 1 (or configured draft/full policy), with analog variability/read noise.
    * $\sigma_{write}$ and readout noise parameters are derived from physical device measurements (**[PLACEHOLDER: Measure data from Song et al. or 1T1R device paper]**).
3. **ReRAM interface precision only:** quantization/dequantization is applied at the ReRAM I/O interface (finite DAC input precision and ADC output precision).
4. **Non-ReRAM compute precision contract:** outside the ReRAM interface, computations remain FP/BF16 (for example SRAM-CIM attention, KV-cache path, softmax, and elementwise digital units).
5. **Residual verification:** verification uses all residual arrays (full analog reconstruction path) and full-precision digital compute assumptions.

#### 5.2.2 Speculation Logic Implementation

The simulator tracks the **Draft Acceptance Rate ($\alpha$)**.

* **Draft Step:** Run forward pass using $W_{draft}$ to propose a draft sequence (a burst of $K$ tokens).
* **Verification (Always On, Pipelined):** For each drafted burst, run a pipelined verification using the full-precision weights ($W$). The verifier always computes **$K+1$ tokens** for that burst (e.g., $K=5 \Rightarrow 6$ pipelined verify steps). If a mismatch occurs early, the remaining verifier steps are still executed (cannot early-stop) but are discarded as wasted work.
* **Commit Rule:** Commit the accepted draft prefix, then commit the verifier’s token at the first mismatch position. If all $K$ draft tokens match, commit all $K$ draft tokens plus the verifier’s bonus token.
* **Acceptance Statistics:** Record the accepted-prefix length per burst ($a \in [0, K]$). For accurate estimator inputs, export a histogram of $a$ (0..$K$), not just a single average. For each burst, **committed output tokens** are $a+1$, while **verifier steps** are always $K+1$ (so wasted verifier steps are $K-a$ when $a<K$).

---

### 5.3 Part B: Hardware Performance Estimator (The Calculator)

We will develop a configurable Python calculator (`selfspec-calculator`) to estimate physical PPA (Power, Performance, Area). This tool inputs the usage statistics from Part A (at minimum, the draft burst length $K$ and accepted-prefix-length statistics) and maps them to physical metrics using a parametric hardware model aligned with the Dual-Mode Residual Architecture described above (Draft Mode on Array 1, Verify Mode using Arrays 2-4 with non-destructive reuse), plus:

* **Analog CIM (ReRAM Residual Arrays):** QKV projection, $W_O$ projection, and FFN (MLP) matmuls.
* **Digital CIM (SRAM, FP/BF16):** attention matmuls ($QK^\top$ and $P V$), where $P=\text{Softmax}(QK^\top)$ is computed by a dedicated softmax unit.
* **Dedicated digital blocks:** KV-cache and a digital processing unit (softmax + elementwise ops such as activation functions and pointwise multiplies).

#### 5.3.1 Inputs (Two YAML Configs + Runtime Speculation Stats)

The estimator is driven by two configuration files plus runtime speculation statistics:

1. **Transformer Model Config (`model.yaml`)**
   * Purpose: Describe the Transformer so we can derive what hardware is needed (matrix shapes, number of layers, and the MAC/read counts implied by attention + MLP).
   * Example fields (illustrative): number of layers, hidden size, number of heads, MLP expansion, vocab size, target sequence length, batch size, and activation/interface precision knobs.
   * FFN type: specify the FFN architecture (e.g., classic 2-projection MLP with GELU, or gated FFN such as SwiGLU with gate/up/down projections) and the corresponding intermediate sizes.
   * Draft precision policy (from §4.1): specify, for each layer, whether the following three **analog matmul** blocks use **draft mode** or **full precision** during Draft Mode:
     * QKV projection (Q/K/V)
     * $W_O$ projection
     * FFN (MLP)
     This allows “sensitivity-aware” precision at submodule granularity.

2. **Hardware Config (`hardware.yaml`)**
   * Purpose: Describe model-irrelevant hardware/technology knobs (crossbar sizing/tiling, ADC precision and type, peripheral energy/latency/area parameters, process assumptions).
   * Example fields (illustrative):
     * **Analog CIM (ReRAM Residual Arrays):** crossbar rows/cols, number of residual arrays (e.g., 1+3 split), DAC input precision, ADC-Draft bit-width and characteristics, ADC-Residual bit-width and characteristics, device read energy/latency, and area models.
     * **Draft Result Reuse in Verify:** whether verification reuses stored draft readout results (e.g., $D_{reg}$) or always re-reads Array 1 for drafted tokens (to study correctness vs PPA when verifier inputs differ from draft inputs).
     * **Digital CIM (SRAM, FP/BF16, per-layer):** attention matmul engine for $QK^\top$ and $P V$ (where $P=\text{Softmax}(QK^\top)$ is computed by a dedicated softmax unit), including array/tile sizing, energy/latency per MAC (or per matmul), and area models.
     * **KV-Cache (Dedicated Digital, per-layer):** memory technology/size assumptions, read/write energy/latency, and area models.
     * **Maximum Context Capacity:** maximum supported context length (KV-cache capacity upper bound) for a full-chip configuration.
     * **Digital Processing Unit (Dedicated Digital, per-layer):** softmax and other elementwise ops (e.g., GELU/SiLU and pointwise multiply), with energy/latency and area models.
     * **Buffers + Add (Per-Layer):** storage and arithmetic overhead for:
       * buffering draft readouts (e.g., $D_{reg}$ or full-precision draft outputs) and storing MVM outputs,
       * combining ADC outputs (and combining draft + correction where reuse is enabled),
       * and digital accumulation/add operations used to form final matmul outputs.
     * **Digital / Control:** accumulation/adder tree costs, registers (e.g., $D_{reg}$), and any controller overhead.

3. **Runtime Speculation Inputs (from Part A / experiment driver)**
   * Draft burst length: **$K$**
   * Acceptance statistic(s): **accepted-prefix-length histogram** $P(a)$ for $a \in [0, K]$ (preferred), plus optional summaries like $\alpha$. The estimator ultimately needs the expected number of **committed output tokens per burst** (which is $\mathbb{E}[a+1]$) to compute throughput and energy/token.
   * Prompt length sweep: a list or range of prompt lengths $L_{prompt}$ to evaluate (bounded by the maximum context capacity).
   * Interpretation: each sweep point represents the **first generated token after the prompt**, so the initial attention/KV context length is $L_{prompt}$. Since a burst computes up to $K+1$ steps, the maximum total context touched within a burst is approximately $L_{prompt}+K$ and must not exceed the maximum context capacity.

#### 5.3.2 Outputs

Given `model.yaml`, `hardware.yaml`, $K$, and acceptance statistic(s), the calculator outputs:

* **Tokens per Second (Throughput)** and overall end-to-end latency, as a function of prompt length $L_{prompt}$ (sweep output).
* **Latency Breakdown** (e.g., Draft Mode vs. Verify Mode vs. setup/overhead).
* **Energy Consumption** and **Energy Breakdown** (e.g., memristor arrays vs. ADCs vs. SRAM digital CIM vs. KV-cache vs. digital processing unit vs. DAC vs. buffers/add vs. digital/control).
* **Area Breakdown** (memristor arrays, ADC-Draft, ADC-Residual, SRAM digital CIM arrays, KV-cache, digital processing unit, DAC, buffers/add, digital, and any SRAM/register overhead).
* **Break-even Prompt Length:** using the **end-to-end cost per committed output token** (including speculative wasted verify work), report the prompt length where attention-related cost (SRAM digital CIM attention matmuls + KV-cache + softmax/attention math) exceeds the context-independent “linear” matmul cost (analog QKV/$W_O$/FFN). Compute this separately for:
  * **Latency break-even:** $L^*_{latency}$
  * **Energy break-even:** $L^*_{energy}$

#### 5.3.3 Accounting Approach (Aligned with the Architecture Above)

Rather than relying on a single closed-form equation upfront, the estimator will use **event-based accounting** that mirrors the Dual-Mode behavior:

* **Draft Compute events (Per Drafted Token):**
  * **Analog Matmuls (ReRAM Residual CIM):** QKV projection, $W_O$ projection, and FFN, using the per-layer draft precision policy.
  * **Attention Matmuls (SRAM Digital CIM, FP/BF16):** compute $QK^\top$ and $P V$ (matmul only).
  * **KV-Cache (Dedicated Digital):** KV-cache accesses.
  * **Digital Processing Unit (Dedicated Digital):** compute $P=\text{Softmax}(QK^\top)$ and perform elementwise ops required by the FFN type (e.g., GELU/SiLU and pointwise multiply).
  * **Buffers + Add:** buffer draft outputs, combine ADC outputs, and perform required digital adds/accumulation.
* **Verify Compute events (Per Verifier Step):**
  * Run full-precision verification for **$K+1$ steps per burst** (cannot early-stop). Only the first $a+1$ steps are logically useful; the remaining $K-a$ steps are wasted but still consume energy/latency.
  * **Attention Matmuls (SRAM Digital CIM, FP/BF16):** always full precision.
  * **KV-Cache (Dedicated Digital):** always full precision.
  * **Digital Processing Unit (Dedicated Digital):** always full precision (softmax + elementwise ops).
  * **Analog Matmuls (ReRAM Residual CIM):**
    * **Reuse enabled:** for verifier steps that correspond to drafted tokens:
      * If the block was executed in default Draft Mode: use non-destructive reuse (activate Arrays 2-4 and fire ADC-Residual, then compute $Final = D_{reg} + C$).
      * If the block was executed in full-precision Draft Mode: reuse the stored full-precision output directly (no additional analog read for that block).
    * **Reuse disabled:** for verifier steps that correspond to drafted tokens, perform a full analog read: activate Arrays 1-4, use both ADCs, then digitally combine.
    * For the **bonus token** step (no stored draft), always perform a full analog read: activate Arrays 1-4, use both ADCs, and digitally combine the results.
  * **Buffers + Add:** buffer/lookup any stored draft readouts (if reuse enabled), combine ADC outputs, and perform the digital adds needed to form final outputs.
* **Burst / Setup effects:** If using Spatial Pipelined Verification (§2.3), setup/bitline charge penalties are accounted for per burst and amortized across the verified tokens in that burst.

**Draft/Verify Scheduling Assumption:** Draft and verify share the same per-layer hardware resources and are modeled as **strictly serialized** at the burst level: compute the Draft burst first, then compute the $K+1$ verifier steps, with no overlap between Draft and Verify phases.

**Precision Contract Summary:** During inference, only ReRAM input/output interface precision is discretized (DAC/ADC). ReRAM weights are represented by residual analog arrays (not bit-plane quantized), and non-ReRAM compute paths (SRAM-CIM + digital units) are modeled in FP/BF16.

**Timing Model:** For each compute step, total latency is modeled as:

$$ T_{step} = T_{analog} + T_{digital} $$

Where $T_{analog}$ captures crossbar readout + ADC behavior, and $T_{digital}$ captures KV-cache, softmax/attention math, and control/accumulation overhead. In particular, attention-related $T_{digital}$ depends on the prompt length $L_{prompt}$. For this project we report a prompt-length sweep at the start of generation (first token after prompt).

**Full-Chip Mapping Assumption:** The chip is designed to store **all model weights** on-chip (no reprogramming of ReRAM during inference), and each Transformer layer has its own compute resources. Layer outputs still have data dependencies, but the full chip can be modeled as a spatial pipeline across layers.

**Pipeline-Level Performance Model:** With per-layer compute blocks, the estimator will model:

* **End-to-end token latency:** approximately the sum of per-layer stage latencies along the forward path (plus any pipeline fill/flush effects).
* **Steady-state throughput (Tokens/s):** bounded by the slowest per-layer stage time (the bottleneck stage) once the layer pipeline is filled. Since attention cost depends on $L_{prompt}$, throughput is reported as a function of prompt length (sweep).
  * **Per-Layer Stage Decomposition (Illustrative):** each layer is modeled as a sequence of stages such as:
  * Analog CIM: QKV projection
  * KV-cache read + Digital CIM: $QK^\top$
  * Digital processing unit: Softmax
  * Digital CIM: $P V$
  * Analog CIM: $W_O$ projection
  * Analog CIM: FFN (MLP) projections, expanded from `ffn_type` (e.g., 2-stage GELU MLP or 3-projection gated FFN)
  * Digital processing unit: FFN elementwise ops (activation + multiply, depending on `ffn_type`)
  Additional elementwise ops and control are captured in the digital overhead terms.

This allows the estimator to produce coherent **throughput**, **latency**, **energy**, and **area** breakdowns while keeping the architecture knobs (ADC split, crossbar sizing, peripheral models) configurable, and while scaling to a **full-chip** estimate based on mapping the model weights to crossbar tiles (from `model.yaml` + `hardware.yaml`).

---

### 5.4 Design Space Exploration (Experiments)

We will use the simulator to sweep across configurable parameters to find the optimal Hardware-Algorithm configuration.

#### Experiment 1: Optimal ADC Resolution Split

We will test different bit-width allocations for the "1+3 Split" to balance draft quality vs. energy.

* **Test Variable:** Resolution of ADC-A (Draft) vs. ADC-B (Residual).
* **Configurations to Sweep:**
    1. ADC-A: **3-bit**, ADC-B: **13-bit** (Lowest Draft Energy, Low Acceptance).
    2. ADC-A: **4-bit**, ADC-B: **12-bit** (Balanced).
    3. ADC-A: **5-bit**, ADC-B: **11-bit** (High Acceptance, Higher Base Energy).
* **Optimization Target:** Maximize Tokens/Joule.

#### Experiment 2: Noise Tolerance & Fine-Tuning Impact

Quantify the benefit of the "Noise-Injection Adaptation" (Algorithm Opt 2).

* **Test Variable:** Analog Write Noise Std Dev ($\sigma$).
* **Comparison:**
  * Baseline Model (Zero-shot).
  * Noise-Tuned Model (Fine-tuned with $\sigma$).
* **Goal:** Demonstrate that Fine-Tuning maintains $>$85% Acceptance Rate even as hardware noise increases (e.g., up to $\sigma = 0.05$).

#### Experiment 3: Layer Sensitivity Map

Determine which layers require "1-Array Draft" vs "2-Array Draft" (Algorithm Opt 1).

* **Procedure:** Bruteforce sweep. Set one layer to "Draft Mode" and measure Perplexity drop.
* **Output:** A heat map of layer sensitivity, establishing the static lookup table for the generic Draft Allocation policy.
