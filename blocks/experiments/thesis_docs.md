
RAG Retrieval Method Comparison for Banking Q&A

 Table of Contents
I.	Introduction	4
1.	Background of Study	4
2.	Problem statement	4
3.	Research gap	5
4.	Research objectives	5
5.	Significance of the study	5
6.	Scope and limitations	6
II.	Literature Review	6
1.	Retrieval-Augmented Generation and Large Language Models	6
2.	Semantic Retrieval and Dense Representations	7
3.	Keyword-Based Retrieval and Classical Information Retrieval Models	7
4.	Evaluation Metrics for Information Retrieval	7
5.	Statistical Significance Testing in Retrieval Evaluation	8
6.	Financial Literacy and Domain Motivation	8
7.	Summary of Literature and Positioning of This Study	8
III.	Methodology	9
1.	Research Design	9
2.	System Architecture	9
a.	Overview	9
b.	Banking Knowledge Base	10
c.	Query Processing and Constraint Detection	11
d.	Retrieval Layer	11
e.	Result Ranking and Top-K Selection	11
f.	Context Construction	11
g.	Response Generation	11
3.	Data Sources and Dataset Construction	12
a.	Data Source	12
b.	Data Cleaning and Normalization	13
c.	Data Embedding and Semantic Indexing	13
4.	Retrieval Strategies	14
a.	Keyword-based retrieval	14
b.	Semantic Retrieval Using Dense Embeddings	15
c.	Hybrid Retrieval Combining Keyword and Semantic Methods	15
d.	Retrieval Parameters	16
5.	LLM/RAG  Prompting	16
a.	LLM Runtime and Model Selection	16
b.	Context Construction for Generation	17
c.	Scope of Evaluation	18
6.	Evaluation Methodology	18
a.	Offline Evaluation Setup	18
b.	Evaluation Metrics	18
7.	Statistical Analysis	21
Paired t-test (Parametric)	22
Cohen’s d (Two Methods)	24
8.	Experiment Procedure and Reproducibility	25
9.	Ethical and Security Considerations	26
10.	Scope and Limitations	27


    Introduction
    Background of Study
In today’s modern economy, banks play a crucial role in safeguarding money and facilitating financial transactions. To use banking services effectively, individuals must understand how these services operate and how to manage their finances wisely. Financial literacy has therefore become an essential capability that empowers individuals to make informed financial decisions and manage their resources responsibly.
In Cambodia, banking services such as loans, savings accounts, fixed deposits, and mobile banking have expanded in recent years, increasing the need for individuals to understand financial products and make informed decisions. However, many customers still face challenges in accessing clear, accurate, and timely information about banking services, which can contribute to poor financial choices and increased exposure to financial risks. Strengthening financial literacy and improving access to reliable financial information are therefore important for individual financial security and sustainable economic development.
According to the report “Key Findings on Financial Inclusion in Cambodia” that published by  World Bank  in April 2025, only a small proportion of Cambodian adults use formal financial services. Many individuals continue to save at home rather than in banks and rely heavily on informal credit, which exposes them to higher financial risk, exploitation, and limited consumer protection.
Financial chatbots have emerged as a practical solution for providing instant guidance and customer support. In Cambodia, banks have begun adopting such technologies. For example, in February 2025, ABA Bank introduced Navi, Cambodia’s first AI-powered virtual banking assistant, which provides 24/7 assistance through the ABA Mobile application in multiple languages. While these systems improve accessibility and convenience for individual customers, they are typically limited to institution-specific information and do not provide comprehensive guidance across multiple banks or product categories. This limitation motivates the need for a domain-specific financial chatbot capable of retrieving and presenting accurate information from authoritative sources, particularly in the banking sector, where precision directly affects customer trust, regulatory compliance, and financial decision-making.
This thesis investigates how Retrieval-Augmented Generation (RAG) can be designed for the banking domain to improve factual accuracy and relevance in chatbot responses. The core challenge lies in developing retrieval strategies suited to banking data, which is sensitive, structured, and institution-specific. By tailoring retrieval and evaluation methods to this context, the study aims to support more reliable, up-to-date, and comprehensive financial guidance for users in Cambodia.
    Problem statement
Despite the expansion of banking services and the adoption of chatbot technologies in Cambodia, customers still face difficulties in obtaining clear, accurate, and comprehensive information about banking products across different financial institutions. Existing bank-provided chatbots operate within isolated information silos and are restricted to the products and services of a single bank, limiting their usefulness for users who wish to compare or understand offerings across multiple banks.
Developing a cross-bank financial chatbot presents significant technical challenges. Banking data is often structured, institution-specific, and inconsistent, with variations in product naming conventions, codes, and formats across banks. As a result, traditional keyword-based retrieval methods are insufficient and may return incomplete or inaccurate information. Inaccurate responses in the banking domain can undermine customer trust, lead to poor financial decisions, and raise regulatory concerns.
Therefore, there is a need to design and rigorously evaluate effective retrieval strategies for a Retrieval-Augmented Generation (RAG)–based banking assistant tailored to the Cambodian context. The central research problem of this thesis is to determine how retrieval methods can be optimized to ensure accurate, relevant, and timely information retrieval from authoritative banking data sources, while maintaining acceptable system performance for real-world usage.
    Research gap
Despite the increasing adoption of chatbots and artificial intelligence in the banking sector, existing implementations in Cambodia are primarily limited to institution-specific customer support systems. Prior research on chatbots and Retrieval-Augmented Generation (RAG) has largely focused on general-domain applications or unstructured knowledge sources, with limited attention to the banking domain, where data is sensitive, structured, and heterogeneous across institutions. Furthermore, there is a lack of empirical evaluation of retrieval strategies tailored to banking product information in emerging markets such as Cambodia. As a result, there is insufficient understanding of how different retrieval methods affect factual accuracy, relevance, and response reliability in cross-bank financial chatbots. This thesis addresses this gap by designing and systematically evaluating retrieval strategies for a RAG-based banking assistant adapted to the Cambodian banking context.
    Research objectives
    To design a Retrieval-Augmented Generation (RAG) architecture tailored to the banking domain in Cambodia.
    To develop and implement retrieval strategies suitable for structured and institution-specific banking data.
    To evaluate the performance of different retrieval methods in terms of factual accuracy, relevance, and response quality.
    To analyze the trade-offs between retrieval accuracy and system response time in a cross-bank financial chatbot.
    To assess the potential of a RAG-based chatbot in improving access to reliable banking information for Cambodian users.
    Significance of the study
This study has significance at multiple levels. Practically, it contributes to the development of a more reliable and informative banking chatbot capable of delivering accurate, up-to-date, and cross-bank financial information, thereby supporting better financial decision-making among Cambodian users. By improving access to trustworthy banking information, the system has the potential to enhance financial literacy and reduce customer exposure to misinformation and financial risk.
Academically, this research extends existing work on Retrieval-Augmented Generation by focusing on a domain characterized by structured, sensitive, and institution-specific data. The findings provide insights into the effectiveness of retrieval strategies in the banking domain and contribute empirical evidence to the growing literature on domain-specific RAG systems.
From a societal perspective, the study supports Cambodia’s broader goals of financial inclusion and digital transformation by exploring how advanced AI systems can be responsibly applied to strengthen consumer access to financial knowledge.

    Scope and limitations
Scope
This study focuses on the design and evaluation of a Retrieval-Augmented Generation–based chatbot for the banking domain in Cambodia. The system covers selected banking products, including loans, savings accounts, fixed deposits, credit cards, and exchange rates, using authoritative data sources from multiple banks. The evaluation emphasizes retrieval performance, factual accuracy, relevance of responses, and system response time. The chatbot is designed as an informational assistant and does not perform transactional operations or provide personalized financial advice.
Limitations
This study has several limitations. First, the chatbot relies on the availability and quality of curated banking data, which may vary across institutions and over time. Second, the system is evaluated using predefined queries and datasets, which may not capture all real-world user behaviors. Third, regulatory, legal, and ethical considerations related to deploying such a system in production environments are discussed conceptually but are not implemented as part of this research. Finally, the study does not assess long-term user adoption or behavioral changes resulting from chatbot usage.













    Literature Review
    Retrieval-Augmented Generation and Large Language Models
Large Language Models (LLMs) have demonstrated strong capabilities in natural language understanding and generation, forming the foundation of modern conversational systems. The transformer architecture, introduced by Vaswani et al. (2017), revolutionized sequence modeling by relying entirely on self-attention mechanisms, enabling efficient learning of long-range dependencies. This architecture underpins most state-of-the-art language models used in contemporary chatbot systems.
Despite their generative strengths, LLMs are prone to factual inaccuracies, particularly when responding to knowledge-intensive queries. To address this limitation, Lewis et al. (2020) proposed Retrieval-Augmented Generation (RAG), a framework that integrates information retrieval with language generation. In RAG, relevant documents are first retrieved from an external knowledge source, and the retrieved content is then used to condition the generated responses. This approach has been shown to improve factual consistency and reduce hallucination in knowledge-intensive natural language processing tasks. RAG is therefore particularly relevant to domains such as banking, where accuracy and reliability are critical.
    Semantic Retrieval and Dense Representations
Traditional keyword-based retrieval methods often struggle with semantic mismatch, where relevant documents do not share exact query terms. To overcome this limitation, dense retrieval methods encode both queries and documents into continuous vector representations, allowing semantic similarity to be measured directly.
Karpukhin et al. (2020) introduced Dense Passage Retrieval (DPR), a bi-encoder architecture that learns dense embeddings for queries and passages using transformer models. DPR demonstrated significant improvements over sparse retrieval methods in open-domain question answering tasks. This work established dense retrieval as a viable alternative to keyword-based approaches, particularly for semantic-heavy queries.
Similarly, Reimers and Gurevych (2019) proposed Sentence-BERT, which adapts BERT using siamese and triplet network structures to generate semantically meaningful sentence embeddings. Sentence-BERT enables efficient similarity computation and is widely used in semantic search and retrieval systems. Its ability to capture contextual semantics makes it suitable for banking-related queries, where users may express the same intent using varied terminology.
    Keyword-Based Retrieval and Classical Information Retrieval Models
Although dense retrieval approaches have gained prominence, keyword-based retrieval remains an important baseline in information retrieval research. Robertson and Zaragoza (2009) presented the probabilistic relevance framework underlying BM25, one of the most widely used keyword-based ranking functions. BM25 relies on term frequency and inverse document frequency to estimate relevance, offering strong performance in scenarios where exact term overlap is important.
Manning, Raghavan, and Schütze (2008) provided a comprehensive foundation for information retrieval, covering indexing, ranking, and evaluation techniques. Their work highlights the strengths and limitations of classical retrieval methods and emphasizes the importance of choosing retrieval models appropriate to the structure and characteristics of the data. In structured and institution-specific domains such as banking, keyword-based methods alone may be insufficient due to inconsistent naming conventions and terminology across institutions.
    Evaluation Metrics for Information Retrieval
Evaluating retrieval effectiveness is essential for comparing different retrieval strategies. Traditional metrics such as precision and recall measure the relevance of retrieved results but do not account for ranking positions. Järvelin and Kekäläinen (2002) introduced Discounted Cumulative Gain (DCG) and its normalized variant NDCG, which reward placing highly relevant documents earlier in the ranked list. NDCG has since become a standard metric in retrieval evaluation, particularly for systems that return ranked results.
Voorhees (1998) highlighted the impact of variability in relevance judgments on retrieval evaluation, demonstrating that evaluation results can vary significantly depending on judgment consistency. This finding underscores the importance of careful experimental design and consistent evaluation protocols when assessing retrieval performance, particularly in domain-specific applications.
    Statistical Significance Testing in Retrieval Evaluation
When comparing retrieval systems, observed performance differences may arise from randomness rather than true improvement. Smucker, Allan, and Carterette (2007) conducted a comprehensive comparison of statistical significance tests for information retrieval evaluation, demonstrating that inappropriate test selection can lead to misleading conclusions. Their work emphasizes the importance of selecting suitable statistical tests, especially when evaluating multiple retrieval strategies on the same set of queries.
Statistical significance testing is therefore critical for validating improvements in retrieval effectiveness within Retrieval-Augmented Generation systems, ensuring that performance gains are both reliable and reproducible.
    Financial Literacy and Domain Motivation
Beyond technical considerations, financial literacy plays a crucial role in enabling individuals to make informed financial decisions. Lusardi and Mitchell (2014) provided extensive empirical evidence demonstrating that individuals with low financial literacy are more likely to make poor financial choices, incur excessive debt, and face higher financial risk. Their work highlights the broader economic and societal implications of limited financial knowledge.
In the context of Cambodia, where formal financial service usage remains limited and banking products vary across institutions, improving access to accurate financial information is particularly important. A domain-specific financial chatbot grounded in reliable retrieval mechanisms has the potential to support financial literacy by providing users with timely, accurate, and accessible banking information.
    Summary of Literature and Positioning of This Study
Existing research demonstrates the effectiveness of Retrieval-Augmented Generation in improving factual accuracy, the advantages of dense retrieval for semantic matching, and the continued relevance of keyword-based baselines. However, prior studies have largely focused on general-domain datasets or unstructured knowledge sources, with limited attention to structured, institution-specific domains such as banking in emerging markets.
This thesis builds upon established retrieval and generation techniques while addressing a clear gap by designing and evaluating retrieval strategies for a RAG-based banking chatbot tailored to the Cambodian context. By integrating semantic and keyword-based retrieval methods and applying rigorous evaluation and statistical testing, this study contributes both practical insights and empirical evidence to the development of reliable domain-specific conversational systems.
























    Methodology

    Research Design

This study adopts a system development and quantitative experimental evaluation research design. The research involves designing a Retrieval-Augmented Generation (RAG)–based banking chatbot system and conducting controlled experiments to evaluate different information retrieval strategies within the system.
There are 3 retrieval strategies that used for the comparison such as:
Keyword-based retrieval
Semantic retrieval using dense embeddings
Hybrid retrieval combining keyword and semantic methods

All strategies are evaluated using the same curated query set, datasets, and configuration, following a repeated-measures experimental design. This ensures that performance differences across strategies are attributable to the retrieval methods rather than data variation. 
Based on the system design, hypotheses are formulated regarding retrieval effectiveness and retrieval latency, which are tested empirically through offline experiments.

    System Architecture
The system is designed using a modular architecture to support controlled experimentation and extensibility. The overall processing pipeline follows this sequence:

    Overview
This section describes the architectural design of the proposed banking question-answering system. The system is implemented as a retrieval-augmented generation (RAG) pipeline that integrates a structured banking knowledge base with a large language model (LLM) to produce context-aware responses to user queries.
The architecture is organized into modular components, including query processing, document retrieval, result ranking, context construction, and response generation. This modular design enables flexible configuration of retrieval mechanisms while maintaining a consistent response generation workflow.
 
Figer:
    Banking Knowledge Base
The system relies on a banking product knowledge base stored in MongoDB. The dataset contains structured and semi-structured information related to banking products such as loans, savings accounts, fixed deposits, credit cards, and exchange rates. Each document includes attributes such as bank identifiers, product names, supported currencies, interest rates, and descriptive text.
To support semantic search functionality, documents are enriched with vector embeddings generated offline using a sentence-transformer model. These embeddings are stored alongside the original documents and are used exclusively during the retrieval stage.

    Query Processing and Constraint Detection
Upon receiving a user query, the system performs product and constraint detection. This step identifies relevant product categories and extracts constraints such as bank names and currencies. Due to inconsistencies in naming conventions across data sources, extracted bank identifiers are normalized to ensure compatibility with the knowledge base schema. The detected constraints are passed to the retrieval component and are used to refine search results where applicable.
    Retrieval Layer
The retrieval layer is responsible for identifying candidate documents from the knowledge base that are relevant to the user query. The system supports multiple retrieval mechanisms, each operating on the same underlying dataset and interfaces.
Retrieval produces a ranked list of candidate documents, each annotated with a relevance score and retrieval metadata. The internal implementation of each retrieval mechanism is encapsulated within this layer, allowing retrieval methods to be interchangeable without affecting downstream components.
    Result Ranking and Top-K Selection
All retrieved documents are sorted based on their associated relevance scores. To control the amount of information provided to the language model, the system selects the Top-K most relevant documents, where K=5.
This Top-K selection balances information coverage with prompt length constraints, ensuring that the response generation component receives concise yet informative context.
    Context Construction
The context construction component transforms the selected Top-K documents into a structured textual representation suitable for LLM consumption. Key attributes such as bank name, product type, interest rate, and currency are extracted and formatted into a consistent context template.
This step serves as an interface between retrieval and generation, ensuring that the language model receives well-structured and relevant information derived from the knowledge base.
    Response Generation
The final response is generated using the Qwen2.5 large language model. The model receives both the original user query and the constructed context as input and produces a natural-language answer.
The response generation process is identical regardless of how the supporting documents were retrieved. As a result, system behavior at this stage depends solely on the quality of the provided context rather than differences in generation logic.



    Data Sources and Dataset Construction

    Data Source
The data sources used in this research were obtained by crawling the official websites of ten local banks in Cambodia. The collected data were categorized into five main product groups: loans, credit cards, savings accounts, fixed deposits, and exchange rates. All scraped data were then loaded into a MongoDB database and stored in JavaScript Object Notation (JSON) format. JSON was selected as the storage format because each bank provides product information with different structures and attributes; therefore, a flexible schema such as JSON allows efficient storage and easier handling of heterogeneous banking data. 
Data that scraped for this study is only the data that published in the bank’s website and No customer-level or transactional data is included.
 




    Data Cleaning and Normalization

To address inconsistencies across different banks and data collections, several preprocessing steps were applied. These include bank code normalization and alias mapping to ensure consistent bank identification, as well as standardization of currency fields across all records. A unified field named bankkey was introduced as a unique identifier to link and reference all banks consistently across collections. In addition, a dedicated collection was created to store standardized bank names along with their corresponding bank keys, enabling centralized management and reliable cross-collection joins.
 

    Data Embedding and Semantic Indexing
To enable semantic retrieval and hybrid retrieval, all banking product documents are enriched with dense vector embeddings prior to experimentation. This embedding process transforms structured and semi-structured banking information into numerical representations that capture semantic meaning, allowing similarity-based retrieval beyond exact keyword matching.
The embedding process is performed offline using a one-time preprocessing script. For each banking product document, multiple relevant fields—such as product name, bank name, interest rates, eligibility criteria, fees, and descriptive text—are concatenated into a single searchable text representation. This approach ensures that both high-level product descriptions and detailed financial attributes contribute to the document’s semantic representation.
Because banking data often contains nested and heterogeneous structures (e.g., interest rate tiers, repayment modes, or currency-specific values), collection-specific normalization is applied prior to embedding. Complex structures such as fixed-deposit interest schedules and loan interest ranges are flattened into readable textual formats. This normalization step improves embedding quality by preserving key semantic information while maintaining a consistent textual representation across collections.
Each searchable text is encoded using a sentence-transformer–based embedding model, producing a fixed-length dense vector. These vectors are stored directly within the corresponding MongoDB documents as an embedding field, alongside the original structured data. Embedding generation is conducted in batches to ensure computational efficiency and consistency.
This offline embedding strategy offers several advantages. First, it significantly reduces query-time latency by avoiding on-the-fly embedding computation for documents. Second, it ensures reproducibility, as all retrieval experiments operate on a fixed, precomputed embedding space. Finally, it allows semantic retrieval methods to be evaluated fairly and consistently across all banking product categories.
By embedding curated banking data in advance, the system enables effective semantic similarity search and supports hybrid retrieval strategies that combine semantic relevance with structured filtering. This embedding layer forms a foundational component of the proposed Retrieval-Augmented Generation (RAG) architecture.


 

Figer:


    Retrieval Strategies

    Keyword-based retrieval

Keyword-based retrieval is a traditional information retrieval approach that matches user queries with documents based on exact or partial word overlap. In this method, retrieval is primarily driven by structured filters and textual matching rather than semantic meaning.
In a banking context, keyword-based retrieval typically relies on fields such as bank code, currency, product name, and other structured attributes stored in the database. When a user submits a query, the system extracts relevant keywords or constraints (e.g., bank name or currency) and applies them as filters to retrieve matching documents. Results are often ranked using simple heuristics, such as the order of matched records or predefined relevance scores.
The main advantage of keyword-based retrieval is its efficiency and precision when exact terminology is used. It performs well for queries with clear constraints, such as specific bank names or currencies. However, this method is sensitive to vocabulary mismatch and cannot effectively handle paraphrased or semantically similar queries. In domains like banking, where product naming conventions vary across institutions, keyword-based retrieval may return incomplete or irrelevant results.
    Semantic Retrieval Using Dense Embeddings
Semantic retrieval addresses the limitations of keyword-based methods by representing text as continuous vector embeddings that capture semantic meaning. Instead of relying on exact word overlap, both user queries and documents are encoded into dense numerical representations using a neural language model.

In this approach, the system first generates an embedding for the user’s query using a sentence-level embedding model. Each document in the database has a precomputed embedding stored alongside its structured fields. Retrieval is performed by computing the cosine similarity between the query embedding and each document embedding, ranking documents based on semantic similarity.
Semantic retrieval is particularly effective for handling paraphrased queries, synonyms, and intent-based questions, which are common in conversational systems. For example, queries such as “low interest personal loan” and “cheap personal loan” can be matched to the same product even if they share few keywords.
The main trade-off of semantic retrieval is computational cost. Generating embeddings and computing similarity scores over many documents increases retrieval latency, especially when full collection scans are required. Additionally, semantic similarity alone may retrieve conceptually related but contextually irrelevant results if domain-specific constraints are not enforced.

    Hybrid Retrieval Combining Keyword and Semantic Methods

Hybrid retrieval combines keyword-based and semantic retrieval techniques to leverage the strengths of both approaches while mitigating their individual weaknesses. In this method, the system executes keyword retrieval and semantic retrieval independently and then merges their results using a weighted scoring mechanism.
Each document receives:
    a keyword-based relevance score, and
    a semantic similarity score.
These scores are combined using a weighted linear function:
Hybrid Score= α∙Semantic Score+(1-α)∙Keyword Score
Where α controls the relative importance of semantic versus keyword relevance.
Hybrid retrieval improves robustness by ensuring that retrieved results satisfy explicit constraints (such as bank or currency) while still benefiting from semantic understanding. This approach is especially suitable for the banking domain, where queries often contain both structured constraints and natural language intent.
The primary limitation of hybrid retrieval is increased computational overhead, as it requires executing both retrieval pipelines. As a result, hybrid retrieval typically incurs higher latency than either keyword-based or semantic retrieval alone. Nevertheless, it often delivers the best overall retrieval effectiveness, making it well suited for applications where accuracy and reliability are prioritized.
To ensure a fair and controlled comparison between retrieval strategies, a fixed set of retrieval parameters is applied consistently across all experiments unless otherwise specified. These parameters are selected based on practical considerations for interactive chatbot usage and common practices in information retrieval research.
    Retrieval Parameters
To ensure a fair and controlled comparison between retrieval strategies, a fixed set of retrieval parameters is applied consistently across all experiments unless otherwise specified. These parameters are selected based on practical considerations for interactive chatbot usage and common practices in information retrieval research.
    LLM/RAG  Prompting
This research adopts a Retrieval-Augmented Generation (RAG) approach to enhance the quality and reliability of responses generated for banking-related queries. The role of the Large Language Model (LLM) in this study is primarily to generate natural-language answers grounded in retrieved documents, rather than relying solely on the model’s internal knowledge.
    LLM Runtime and Model Selection
The LLM is deployed using a local runtime environment based on Ollama, enabling offline execution and full control over the model behavior and parameters. A lightweight open-source language model (e.g., Mistral or LLaMA-based variants) is used for the generation component.
The use of a local LLM runtime is justified for the following reasons:
    Reproducibility: Local execution ensures consistent results across experiments and repeated evaluations.
    Cost efficiency: The system avoids API usage costs, making it suitable for academic research and repeated testing.
    Flexibility: Model parameters, prompt templates, and inference settings can be easily adjusted during experimentation.
It should be noted that the core focus of this research is on retrieval performance rather than language generation quality; therefore, the LLM serves as a supporting component rather than the primary evaluation target.
    Context Construction for Generation
For each user query, the system first performs document retrieval using one of the evaluated strategies (keyword-based, semantic, or hybrid retrieval). The top-K retrieved documents are then used to construct the context for the LLM.
The context construction process follows these steps:
    Top-K Selection: The highest-ranked documents from the retrieval stage (e.g., Top-5 or Top-10) are selected.
    Field Extraction: Only relevant structured fields are included in the context, depending on the collection type, such as:
    Bank name and bank code
    Product name or loan type
    Currency
    Interest rate or key financial attributes
    Short product description
    Context Formatting: Retrieved documents are concatenated into a structured and readable format to ensure clarity and consistency for the LLM input.
This approach ensures that the LLM receives concise, domain-specific, and relevant information while avoiding unnecessary noise from raw or unrelated fields.
Hallucination Control and Answer Constraints
To reduce the risk of hallucination and unsupported responses, several control mechanisms are applied at the prompting level:
    Context-restricted prompting: The LLM is explicitly instructed to generate answers only based on the provided retrieved context.
    No external knowledge assumption: The model is instructed not to infer or introduce information that does not appear in the context.
    Fallback behavior: If the retrieved documents do not contain sufficient information to answer a query, the model is required to respond with a predefined message indicating that the information is “not found in the retrieved data.”
These controls are critical in the banking domain, where inaccurate or fabricated information could mislead users and compromise trust in the system.
    Scope of Evaluation
Although the system supports full RAG-based answer generation, the primary evaluation in this study focuses on the retrieval component. Metrics such as Precision@K, Recall@K, and related ranking measures are used to compare retrieval strategies independently of the LLM’s language generation capabilities.
As a result, the generation component is included mainly to demonstrate end-to-end system feasibility and practical applicability, while its influence on final answer quality is not quantitatively evaluated in depth.

    Evaluation Methodology

This study evaluates the effectiveness of different information retrieval strategies within the proposed banking chatbot system using a controlled offline experimental setup. The evaluation focuses on comparing retrieval performance across multiple methods under identical conditions to ensure fairness and statistical validity.
    Offline Evaluation Setup
An offline evaluation framework is adopted to enable systematic and reproducible comparison of retrieval methods. The same curated query set is applied consistently to each retrieval strategy, including:
    Keyword-based retrieval
    Semantic retrieval using dense embeddings
Hybrid retrieval combining keyword and semantic approaches
A repeated-measures experimental design is used, where each query is evaluated across all retrieval methods. This design minimizes variance caused by query difficulty and ensures that performance differences can be attributed directly to the retrieval strategy rather than the query set itself.
All retrieval methods operate on the same underlying document collections, indexed data, and preprocessing pipeline, ensuring consistent experimental conditions.
    Evaluation Metrics
Retrieval performance is assessed using standard information retrieval metrics, selected to align directly with the implemented evaluation code and the objectives of the study:

Precision@K
Measures the proportion of retrieved documents in the top-K results that are relevant. Or in simple it is the number of relevant documents retrieved in the top-K divided by K.
    Let Q={q_1,q_2,…,R_|Q|  } is the set of the queries and |Q|   is the number of queries
    Let  R_k={R_k,R_k,…,R_k } is the set of the top-K retrieved documents
    G is the set of the relevant documents
    1(∙)  is the indicator function which defined as 
1(x)={█(1,if x is true@0,otherwise)┤   
Then the formula of the Precision@K defined as
Precision@K=1/K ∑_(i=1)^K▒1(d_i∈G) 
or
Precision@K=|R_k∩G|/K
Please note that the value of the Precision@K is between 0 and 1 or   
0≤ Precision@K≤1

Recall@K
Measures the proportion of relevant documents that are successfully retrieved within the top-K results. Or in simple it is the fraction of all relevant documents that appear in the top-K results.
Recall@K=1/|G|  ∑_(i=1)^K▒1(d_i∈G) 
or
Recall@K=|R_k∩G|/G
The recall@K also having value between 0 and 1 or 
0≤ Recall@K≤1


F1@K
The harmonic mean of Precision@K and Recall@K, providing a balanced performance measure. Or How well the system balances retrieval accuracy and coverage within the top-K results.
F1@K=2*(Precision@K*Recall@K)/(Grecision@K+Recall@K)
or
F1@K=2|R_k∩G|/(K+G)


Mean Reciprocal Rank (MRR)
Evaluates the rank position of the first relevant document in the result list.
                    Recall that 
Q={q_1,q_2,…,R_|Q|  }
|Q|  is the number of queries
Then 
MMR=1/|Q|  ∑_(q∈Q)▒〖RR(q)〗
While Reciprocal Rank (RR) calculated as
RR(q)={█(1/〖rank〗_q ,if 〖rank〗_q<∞@0,otherwise)┤





Normalized Discounted Cumulative Gain (NDCG@K)
Assesses ranking quality by considering both relevance and position of relevant documents within the top-K results.
Unlike Precision or MRR, nDCG supports graded relevance.
For a query q, define a relevance score:  rel(q_i )∈{0,1,2,…}
Where the higher value indicate the higher relevance. 
    Where R_k={R_k,R_k,…,R_k } is the list of retrieved documents.
    The Discounted Cumulative Gain (DCG@K) is defined as:
                DCG@K= ∑_(i=1)^K▒(2^(rel(q_i))-1)/log_2⁡〖(i+1)〗 
To normalize the score, the Ideal Discounted Cumulative Gain (IDCG@K) is computed by sorting the retrieved documents in descending order of relevance to obtain the ideal ranking:
IDCG@K=∑_(i=1)^K▒(2^(" " rel(q,d_i))-1)/(〖log⁡〗_2 (i+1))
    Then the Normalized Discounted Cumulative Gain is calculated as
                    IDCG@K=(DCG@K)/(DCG@K)
The resulting value lies in the range [0ⓜ,1], where a value of 1 indicates a perfect ranking. NDCG@K is particularly effective for evaluating retrieval systems that aim to return highly relevant documents at top positions, such as information retrieval and Retrieval-Augmented Generation (RAG) systems.

Latency (milliseconds)
Latency measures the time required by the retrieval system to process a query and produce a ranked list of results. For a given query q, latency is defined as the elapsed time between the moment the query is submitted to the system and the moment the retrieval output is returned.
Let  t_q^start denote the timestamp when query qis issued,
t_q^end  denote the timestamp when the ranked results are returned.
Then the latency for query q is then defined as:
        〖Latency = t〗_q^end- t_q^start  
With the set of queries Q={q_1,q_2,…,R_|Q|  }
Then the average latency defined as:
        (Latency) ̅=1/|Q|  ∑_(i=1)^(|Q|)▒〖t_(q_i)^end- t〗_(q_i)^start 

Ground truth definition
How “relevant_docs” are defined(bank_code + product name ID)
For the set of queries Q={q_1,q_2,…,R_|Q|  }
D if the set of documents in the corpus. Where d∈Q is uniquely identified by a composite identifier
doc_id(d) = (bank_code(d), product_name(d))
    ground truth relevant set
    For each query  q∈Q, define the ground truth relevant document set as:
For each query q∈Q, define the ground truth relevant document set as:
G(q)={d∈D∣"doc_id"(d)∈R(q)}

where
R(q)={(〖"bank_code" 〗_j,〖"product_name" 〗_j)"  " |"  " j=1,2,…,n_q }
    Relevance function
The relevance of a retrieved document dwith respect to query qis defined by a binary relevance function:

rel(q,d)={■(1,&"if doc_id" (d)∈R(q)@0,&"otherwise" )┤

    Statistical Analysis

Observed differences in retrieval performance metrics between methods may arise due to random variation in query difficulty rather than genuine differences in retrieval effectiveness. Therefore, statistical hypothesis testing is required to determine whether the observed performance differences are statistically significant and unlikely to have occurred by chance.
    
Let    X_i^((A) ),X_i^((B) )
denote the evaluation metric scores (e.g., NDCG@K, MRR) obtained for the same query q_i using retrieval methods Aand B. Statistical testing assesses whether the difference between methods across all queries reflects a true performance difference.

Normality Test (Shapiro–Wilk Test)
Before selecting an appropriate significance test, the normality of the paired score differences is examined using the Shapiro–Wilk test.
Let:
d_i=X_i^((A) )-X_i^((B) ),i=1,…,N

The null hypothesis is:
H_0:{d_i}" are drawn from a normal distribution"

The Shapiro–Wilk test statistic is defined as:
W=(∑_(i=1)^N▒a_i  d_((i) ) )^2/(∑_(i=1)^N▒( d_i-d ˉ)^2 )

where:
    d_((i) )are the ordered differences,
    a_iare constants derived from the covariance matrix of order statistics,
    d ˉis the sample mean.
If p<α, normality is rejected and a non-parametric test is used.
Two-Method Comparisons (Paired Tests)
Since all retrieval methods are evaluated on the same set of queries, paired statistical tests are applied.
Paired t-test (Parametric)
Used when the normality assumption holds.
Null hypothesis:
H_0:μ_d=0

Test statistic:
t=d ˉ/(s_d/√N)

where:
    d ˉis the mean of the paired differences,
    s_dis the standard deviation of the differences,
    Nis the number of queries.
Wilcoxon Signed-Rank Test (Non-Parametric)
Used when normality is violated.
Let r_ibe the ranks of ∣d_i∣(excluding zeros). The test statistic is
W=∑"signed ranks" 
Null hypothesis
H_0:"median"(d_i)=0

This test does not assume normality and is robust to outliers.

Three or More Method Comparisons
Friedman Test (Non-Parametric:
Used to compare three or more retrieval methods under repeated-measures conditions.
Let:
    kbe the number of methods,
    Nbe the number of queries,
    R_(i,j)be the rank of method jfor query i.
The Friedman test statistic is:
χ_F^2=12N/(k(k+1)) (∑_(j=1)^k▒R ˉ_j^2 )-3N(k+1)

where R ˉ_jis the mean rank of method j.
Repeated-Measures ANOVA (Parametric)
When normality and sphericity assumptions are met, repeated-measures ANOVA is used.
The F-statistic is defined as:
F="Between-method variance" /"Within-method variance" 
Effect Size Measures
Statistical significance alone does not indicate practical importance. Effect sizes are reported to quantify the magnitude of differences.
Cohen’s d (Two Methods)

(d=d ̅/s_d )


Interpretation:
d=0.2: small effect
d=0.5: medium effect
d=0.8: large effect
Eta-Squared (η²) (Multiple Methods)

(η^2=(SS_"between" )/(SS_"total"  ))

where SSdenotes sum of squares.

Multiple Comparison Correction
When conducting multiple pairwise tests, the risk of Type I error increases. Corrections are applied.

Bonferroni Correction
α^'=α/m
where m is the number of comparisons.

Holm–Bonferroni Correction

Adjusted p-values are computed as:
p_i^'=(max⁡)┬(j≤i) ((m-j+1)p_((j) ) )
This method controls the family-wise error rate while being less conservative than Bonferroni.

Significance Level and Decision Rules
All hypothesis tests are conducted at a significance level of:
(α=0.05)
Decision rules:
    If p<α: reject H_0, difference is statistically significant
    If p≥α: fail to reject H_0
Statistical conclusions are reported together with effect sizes to ensure both statistical and practical relevance.
    Experiment Procedure and Reproducibility
To ensure a systematic and reproducible evaluation, all experiments follow a fixed and deterministic workflow:

    Load Configuration
All experiment parameters are loaded from a configuration file, including retrieval strategies, evaluation metrics, cutoff rank K, and paths to datasets. This ensures that the experimental setup remains consistent across runs.
    Load Curated Test Queries
A manually curated set of evaluation queries is loaded. Each query includes predefined ground truth relevance labels based on composite document identifiers ("bank_code" ⓜ,"product_name" ).
    Run Retrieval for Each Strategy and Query
For each query q_iand each retrieval strategy m∈{"keyword","semantic","hybrid"}, the retrieval module returns a ranked list of top-K documents from the corpus.
    Compute Evaluation Metrics
Retrieval results are evaluated against the ground truth using standard information-retrieval metrics, including Precision@K, Recall@K, F1@K, MRR, and NDCG@K. Latency is measured for each query and strategy.
    Run Statistical Tests
Metric values are aggregated across queries and subjected to statistical hypothesis testing under a repeated-measures design. Normality is assessed, followed by appropriate paired or multi-method significance tests and effect-size calculations.
    Generate Tables and Plots
Final results are summarized in tables and visualized using plots (e.g., boxplots or bar charts) to facilitate comparison across retrieval strategies.
 



    Ethical and Security Considerations
This study adheres to ethical and security best practices relevant to banking information systems. All data used in the experiments were obtained from publicly available or institutionally approved sources and consist solely of product-level information (e.g., interest rates, fees, and product descriptions). No personal customer data, transactional records, or personally identifiable information (PII) were collected, stored, or processed at any stage of the system design or evaluation.
The developed RAG-based system is intended strictly for informational and decision-support purposes. It does not provide personalized financial advice, execute financial transactions, or interact with live banking systems. The system’s outputs are designed to assist users in understanding available banking products and should not be interpreted as professional financial recommendations.
These constraints help minimize ethical risks, ensure data privacy, and align the system with responsible AI usage in the financial domain.
    Scope and Limitations

Scope of the Study
This study focuses on evaluating a Retrieval-Augmented Generation (RAG) system using structured banking product information collected from selected commercial banks operating in Cambodia. The evaluation covers multiple product categories, including loans, savings accounts, fixed deposits, credit cards, and exchange rates. Only banks for which publicly available and consistently structured product information was accessible at the time of data collection were included.
The system supports queries in English, reflecting the primary languages used by customers in the Cambodian banking context. Retrieval and evaluation are conducted within this bilingual setting using the same underlying corpus and retrieval strategies.
Limitations
The dataset used in this study represents a snapshot in time and may not fully reflect the most recent changes to banking products, interest rates, or fees. As a result, retrieval performance is evaluated on static data rather than continuously updated information. This limitation may affect the applicability of the system in real-time deployment scenarios without regular data refresh mechanisms.
In addition, the system is evaluated in an offline experimental setting, and operational constraints such as system scalability, concurrent user handling, and integration with production banking infrastructure are outside the scope of this research.


















    Implementation

    Implementation Overview
This section describes the concrete implementation of the proposed banking RAG system. The implementation is organized as modular Python components for (1) data preparation and embedding, (2) query understanding, (3) multi-strategy retrieval, (4) context construction and LLM response generation, and (5) offline experimentation and statistical analysis.

The end-to-end runtime flow is:
User query → constraint extraction (intent, bank, currency) → retrieval (`keyword`, `semantic`, or `hybrid`) → Top-K context construction → LLM answer generation.

    Core Application Implementation
The main application logic is implemented in `blocks/main/main.py`.

Key implemented components include:
- **Configuration and connectivity**: default Ollama endpoint `http://127.0.0.1:11434`, default model `qwen2.5:3b`, and MongoDB database `banking_db`.
- **Collection-level access**: dedicated collection mappings for `bank_code`, `exchange_rates`, `fixed_deposits`, `savings_accounts`, `loan`, and `credit_cards`.
- **Query preprocessing**:
  - `extract_bank_codes_from_text()` performs bank name/code matching with token-based and acronym-based heuristics.
  - `parse_currency_hint()` detects major currency hints (USD/KHR).
  - `split_user_question()` supports multi-question decomposition.
  - `classify_intents()` routes user questions by domain intent.
- **LLM integration**: `ollama_chat()` supports multiple API-compatible endpoints (`/api/generate`, `/api/chat`, `/v1/chat/completions`) and performs endpoint probing for robustness.

    Retrieval Engine Implementation
All retrieval strategies are implemented in `blocks/main/retrieval_strategies.py` and exposed through the unified function `retrieve()`.

    Keyword Retrieval
`keyword_retrieval()` uses MongoDB queries with structured filters and ranking metadata.
- Supports bank and currency filtering.
- Applies fallback bank extraction (`extract_bank_codes_fallback()`) if no bank code is detected upstream.
- Resolves inconsistent bank identifiers with `normalize_bank_codes()` and `BANK_CODE_NORMALIZATION`.

    Semantic Retrieval
`semantic_retrieval()` uses dense vector similarity.
- Query embedding is generated using `SentenceTransformer`.
- Documents are loaded from records containing an `embedding` field.
- Cosine similarity is computed by `compute_cosine_similarity()`.
- Default similarity threshold is `0.3`; only documents above threshold are kept.
- Optional bank/currency post-filtering is applied after semantic ranking.

    Hybrid Retrieval
`hybrid_retrieval()` combines keyword and semantic outputs.
- Default blending parameter is `\alpha = 0.5`.
- Implemented score fusion is:
$$
	ext{HybridScore} = \alpha \cdot \text{SemanticScore} + (1-\alpha) \cdot \text{KeywordScore}
$$
- Final results are sorted by fused score and ranked before Top-K truncation.

    Embedding and Index Preparation
Embedding preparation is implemented in `blocks/data_integrations/add_embeddings.py`.

Implementation details:
- The script processes multiple banking collections via a predefined `COLLECTIONS` field map.
- `create_searchable_text()` flattens structured and nested fields into a normalized text representation.
- Collection-specific flattening is implemented for complex rate structures:
  - `flatten_interest_rates()` for fixed deposits.
  - `flatten_loan_rates()` for loan products.
- `add_embeddings_to_collection()` encodes documents in batches and writes both:
  - `searchable_text`
  - `embedding`
back into MongoDB documents.

This implementation supports reproducible offline indexing and avoids document-time embedding generation during retrieval.

    Comparison Chatbot Implementation
The interactive side-by-side comparison interface is implemented in `blocks/main/comparison_chatbot.py`.

Implemented behavior:
- Runs all three strategies (`keyword`, `semantic`, `hybrid`) on the same user question.
- Displays retrieved products, answer text, and timing for each strategy.
- Captures human ratings and preferred method.
- Appends evaluation logs to `comparison_logs.jsonl` for later analysis.

    Experimental Pipeline Implementation
The automated experiment workflow is implemented in `blocks/experiments/run_experiments.py`.

Pipeline steps implemented in code:
1. Load `ExperimentConfig`.
2. Load curated test queries (including collection-specific JSON files).
3. Execute each query across configured retrieval strategies.
4. Record retrieval outputs, timing, and metadata.
5. Save raw outputs and intermediate checkpoints.
6. Compute summary metrics and export analysis artifacts.

Generated artifacts include:
- `experiment_results_raw.json`
- `experiment_results_intermediate.json`
- `experiment_metrics.csv`
- `statistical_analysis_results/` (tables, plots, and JSON summaries)

    Evaluation Metrics Implementation
Metric computation is implemented in `blocks/experiments/evaluation_framework.py`.

The evaluator implements:
- `precision_at_k()`
- `recall_at_k()`
- `f1_at_k()`
- `mean_reciprocal_rank()`
- `ndcg_at_k()`
- query-level `latency_ms`

Ground-truth matching is performed using a composite document identifier produced by `_doc_id()` (bank + product identifier fields).

    Statistical Analysis Implementation
Statistical analysis is implemented in `blocks/experiments/statistical_analysis.py`.

Implemented tests and analysis utilities include:
- Normality test: Shapiro–Wilk (`test_normality()`)
- Variance homogeneity test: Levene (`test_homogeneity()`)
- Paired parametric comparison: `paired_t_test()`
- Non-parametric paired comparison: `wilcoxon_signed_rank()`
- Multi-method repeated-measures comparison: Friedman (`repeated_measures_anova()` method using Friedman statistic)
- Post-hoc pairwise testing with multiple-comparison correction (`bonferroni` and `holm` options)

The statistical analyzer also exports publication-oriented outputs (CSV summaries, visualization figures, and structured JSON reports).

    Implementation Notes on Robustness
Based on the implemented codebase, robustness is handled through several practical mechanisms:
- Bank alias expansion and bank-code normalization to reduce cross-collection mismatches.
- Multi-endpoint Ollama compatibility with diagnostic probing.
- Structured fallback behaviors for empty retrieval or unavailable resources.
- Repeated-measures experiment execution for consistent cross-strategy comparison.

Overall, the implementation provides a complete and reproducible prototype for comparing retrieval strategies in a Cambodian banking RAG assistant.







        





