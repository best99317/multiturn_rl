# Intern-Project Report: Confident Multi-turn Conversational Recommender with Informative Interactions 

Intern: [Cedar Bai](https://quip-amazon.com/EHV9EAvALWn)
Mentor: [Duanshun Li](https://quip-amazon.com/TZA9EAUEEVq) [Sheikh Sarwar](https://quip-amazon.com/IIX9EAnjidR) [Zhenyu Liao](https://quip-amazon.com/dRV9EAazRw3)

## Abstract

Recent advances in large language models (LLMs) have enabled their use as conversational recommender systems (CRS), demonstrating strong performance in both recommendation accuracy and natural dialogue. However, guiding multi-turn interactions to effectively elicit user preferences remains challenging. Existing approaches either use separate reinforcement learning agents with templated interactions, or optimize for interactivity judged by another LLM—without measuring how much useful information is actually gained. We propose a new approach that quantifies the effectiveness of each interaction by the reduction in the assistant’s uncertainty, measured via entropy over recommendations. We apply this entropy reduction as a reward—without relying on ground-truth recommendations, which are often untractable in real-world scenarios—to fine-tune the LLM, enabling strategic interaction generation. Empirical results with supervised fine-tuning (SFT) and direct preference optimization (DPO) on the Inspired and ReDial datasets show our method improves both recommendation quality and conversation efficiency. 


## 1. Introduction

Conversational recommendation systems (CRS) ([Sun & Zhang, 2018](https://arxiv.org/pdf/1806.03277); [Li et al., 2018](https://arxiv.org/pdf/1812.07617); [Chen et al., 2019](https://aclanthology.org/D19-1189.pdf); [Zhou et al., 2020](https://arxiv.org/pdf/2007.04032); [Wang et al., 2022](https://arxiv.org/pdf/2206.09363)) have recently garnered increasing attention due to their ability to interpret user intent more effectively and provide personalized recommendations through natural language interaction. These systems engage users in multi-turn dialogues to dynamically elicit preferences, clarify ambiguous requests, and iteratively refine recommendations based on real-time feedback, fundamentally transforming recommendation from a one-shot prediction task into an interactive preference discovery process.

With the rapid development of large language models (LLMs), there has been a noticeable shift toward employing LLMs directly as conversational recommenders ([He et al., 2023](https://arxiv.org/pdf/2308.10053); [Zhu et al., 2024](https://dl.acm.org/doi/pdf/10.1145/3589334.3645347); [2025](https://dl.acm.org/doi/pdf/10.1145/3696410.3714908); [Zhang et al., 2025](https://arxiv.org/pdf/2310.19488); [He et al., 2025](https://dl.acm.org/doi/pdf/10.1145/3701551.3703573)), demonstrating superior performance over conventional systems ([Li et al., 2018](https://arxiv.org/pdf/1812.07617); [Chen et al., 2019](https://aclanthology.org/D19-1189.pdf); [Zhou et al., 2020](https://arxiv.org/pdf/2007.04032)) in both delivering accurate recommendations and conducting user-aware dialogues.

Figure 1. Example of conversational recommendation

Despite this progress, key challenges remain in this field, including understanding and eliciting user preferences through multi-turn conversations, integrating user history and collaborative filtering information, retrieving up-to-date recommendation candidates, grounding LLM recommendations in real-world knowledge, etc. In this project, we focus on the **first challenge**—understanding and eliciting user preferences through strategic, multi-turn interactions—while leaving the others for future exploration.

We specifically examine how to design interactions that most effectively solicit user preferences. In an ideal setting, conversational recommendation should be a **collaborative process** between the user and the AI assistant, rather than a one-way delivery of suggestions. 

Existing approaches that leverage multi-turn dialogue to capture user preferences typically involve training a separate reinforcement learning (RL) agent to decide when and what to ask or recommend ([Deng et al., 2021](https://arxiv.org/pdf/2105.09710); [Du et al., 2025](https://aclanthology.org/2025.naacl-long.133.pdf)). In such settings, LLMs are used only for dialogue generation, not recommendation, and interaction styles are restricted to rigid formats such as multiple-choice questions or Yes/No responses—limiting natural conversational flow.


Figure 2. Coversational recommendation formulated as decision making between when and what to ask or recommend in formatted multiple-choice questions and Yes/No responses ([Deng et al., 2021](https://arxiv.org/pdf/2105.09710); [Du et al., 2025](https://aclanthology.org/2025.naacl-long.133.pdf)).

In parallel, recent work CollabLLM has encouraged assistant interactivity by fine-tuning LLMs with rewards judged by another LLM ([Wu et al., 2025](https://arxiv.org/pdf/2502.00640)). While effective in making LLMs more proactive in general-purpose tasks (e.g., math problem solving, essay writing), in recommendation contexts it remains unclear how much useful preference information is actually gained from these interactions and how they would benefit conversational recommendation.

Figure 3. Prompts used by CollabLLM ([Wu et al., 2025](https://arxiv.org/pdf/2502.00640)) to generate reward that evaluates the AI assistant’s interactivity. A is set to 1.0, B is 0.5, and C is 0.

To tackle this challenge, we draw inspiration from recent findings on the correlation between recommendation reliability and uncertainty by [Kweon et al., 2025](https://arxiv.org/pdf/2501.17630). They quantify LLM recommendation uncertainty through predictive uncertainty, measured by whether the LLM can consistently produce identical recommendations across multiple generations. In the ranking setting, they propose to repeatedly sample the top-1 recommendation and compute the entropy across generations. Their key observation is that predictive uncertainty indicates reliability of recommendation, in other words, a recommendation with lower uncertainty yields higher recommendation performance.

In this project, we propose a method to quantify the effectiveness of each assistant interaction in the context of conversational recommendation by the information provided in the corresponding user response. We measure this information gain as the reduction in the assistant’s recommendation uncertainty, where uncertainty is computed via the entropy over recommendations. This entropy reduction is then used as a reward—without relying on ground-truth recommendations, which are often unavailable in real-world scenarios—to fine-tune the LLM for strategic interaction generation. Empirical results on the Inspired and ReDial datasets, using both supervised fine-tuning (SFT) and direct preference optimization (DPO), demonstrate that our method improves both recommendation quality and conversational efficiency.

### Contributions

1. We introduce a reward metric based on reduction in recommender uncertainty, which simultaneously improves conversation strategy and recommendation accuracy.
2. The metric is particularly valuable when ground-truth recommendations are unavailable, providing a practical solution for real-world CRS.
3. We contribute two turn-level completion-pair datasets generated from the Inspired and ReDial datasets, designed for DPO fine-tuning in conversational recommendation.

## 2. Proposed Method

In conversational recommendation, the recommender (or AI assistant) engages users in multi-turn dialogues to dynamically elicit preferences, often by asking targeted questions. With each turn, the user provides more information, enabling the assistant to better understand their preferences and become more certain about what to recommend. For example, at the very beginning, the assistant has no prior knowledge of the user and is highly uncertain, resorting to essentially random recommendations. As the dialogue unfolds, the user reveals likes and dislikes, recently watched movies, and other relevant details. To encourage the assistant to ask the most informative questions, we aim for each interaction to significantly reduce recommendation uncertainty. We now introduce our measure of uncertainty.

### 2.1 Uncertainty Measurement

We define uncertainty as the entropy over recommendations. Specifically, we use the following prompt to sample lists of recommendations based on a given conversation C.


>Given the following conversation history:

{conversation C}

Generate a list of the top {m} movies the user would like to watch based on this conversation. Format your response as a numbered list with no extra sentences:

1. [First movie]
2. [Second movie]
3. [Third movie]

Make sure each recommendation is unqiue and plausible given the conversation context.



#### **2.1.1 Entropy over recommendations**

We sample n lists {L(i)}i=1n​ generated by the AI-assistant based on C, each list L(i)containing top-m movie recommendations L(i)=[rj(i)​]j=1m​. The generated lists form a set of recommendations R=Set({L(i)}i=1n​). For each item r∈R, its count 

c(r)=# of times r appears in {L(i)}i=1n​

and its frequency is
p(r)=m×nc(r)​

which serves as an empirical estimation to recommendation distribution. The entropy over recommendations is calculated as 

H(C)=−∑r∈R​p(r)log2​p(r)


#### **2.1.2 Weighted entropy over recommendations**

To further incorporate the difference in ranking order in each recommendation list L(i)=[rj(i)​]j=1m​, we apply the following weighted entropy with logarithmic decay. For each rank position j∈{1,2,...,m}, the weight

wj​=log2​(j+1)1​.

For each item r∈R, its weighted count is defined as

cw​(r)=∑i=1n​∑j=1m​wj​⋅1{rji​=r}​

where 1{rji​=r}​ is the indicator function that equals 1 if rji​=rand 0 otherwise. The weighted probability distribution is calculated as
pw​(r)=∑r′∈R​cw​(r′)cw​(r)​=n×∑j=1m​wj​cw​(r)​,

and the weighted entropy is calculated as 

Hw​(C)=−∑r∈R​pw​(r)log2​pw​(r)


### 2.2 Information Gain, i.e., Uncertainty Reduction of an Interaction

With uncertainty defined, we measure the reduction in uncertainty after an interaction, which we regard as the information gained for recommendation.

#### **2.1.1 Turn** entropy reduction

* Given a partial conversation, e.g., coontaining the initial user query C1​=[U1​], the uncertainty of the recommender can be calculated with Hw​(C1​). 
* Now an assistant interaction A1​is generated, e.g., an initial recommendation or a question that asks for more specific user preferences. 
* The user then responds to A1​ with U2​, e.g., like or dislike the initial recommendation or inform the recommender of their preferences. 
* Now we have an extended conversation C2​=[U1​,A1​,U2​]. The uncertainty is then given by Hw​(C2​). 
* The uncertainty reduced or the information gained by turn A1​ is defined as 

IT​(A1​)=Hw​(C1​)−Hw​(C2​). 

#### **2.1.2 Conversation** entropy reduction

We also propose a variant of turn-level entropy reduction, called conversation entropy reduction, which measures a turn’s effectiveness not only by the uncertainty it directly reduces, but by the total reduction achieved over the full conversation unrolled from that turn. Specifically,

* Given a partial conversation, e.g., C1​=[U1​], calculate Hw​(C1​) as before. 
* An assistant interaction A1​is generated.
* Unroll the full conversation from A1​till conversation terminated, i.e., C=[U1​,A1​,U2​,A2​,...,Aτ​,Uτ​]. The uncertainty is then given by Hw​(C). 
* The conversation entropy reduction by turn A1​ is defined as 

IC​(A1​)=Hw​(C1​)−Hw​(C). 

### 2.3 An Illustrative Example

For following the conversation from the Inspired dataset ([Hayati et al., 2020](https://arxiv.org/pdf/2009.14306)) generated by human workers,


>**Assistant**: Hi! I'm here to help you chose a movie!
**User**: Terrific
**Assistant**: What are some genres you like? What was the last movie you saw?
**User**: The last movie i saw in the theater was Hustlers. I generally like comedy, drama and documentaries
**Assistant**: How did you like Hustlers? It definitely has the drama aspect, did it leave you wanting more or was it not exactly what you were looking for?
**User**: I liked it,  it wasn't the most high-brow movie i've ever seen but it was fun. I like to just enjoy
**Assistant**: And what are your thoughts on action in film?
**User**: I like some action. Just not too violent
**Assistant**: Definitely, movies like Deadpool wouldn't be up your alley I'm guessing?
**User**: It could be. I understand Deadpool was very funny  I like funny
**Assistant**: Haha it definitely was, I think I have a suggestion for you though
**User**: Cool  Whats that?
**Assistant**: So it's the new Beautiful Day in the Neighborhood, the Mr. Rogers movie with Tom Hanks as the actor. It's not as much a comedic relief, but I saw it in theatres with my roomates and it's honestly a super refreshing and incredibly well directed movie.
**User**: that sounds so good to me. I loved Mr. Rogers so much.. and Tom Hanks is a great actor
**Assistant**: The combination is incredible to be honest, and it lets you understand his background and who he was as a person so much more
**User**: That sounds amazing. I would love to see that

 
after each user response, we sample n=5 lists of top-m=5 movie recommendations and calculate the corresponding entropy. Indeed, as shown in Figure 4., entropy exhibits a decreasing trend as the conversation progresses and the assistant gathers more information from the user.

Figure 4. Recommendation entropy calculated after each user response.


### 2.4 Fine-tuning with Entropy Reduction

We propose to fine-tune the LLMs with the proposed entropy reduction as reward. For SFT, we select the training examples with high reward. For DPO, among two generations sampled from the recommender, the one with high reward is preferred.

## 3. Experiments

### **3.1 Baselines**

* Vanilla: We compare with the vanilla zero-shot recommender ([He et al., 2023](https://arxiv.org/pdf/2308.10053)), i.e., directly use the pre-trained LLM as the recommender.
* We compare with the LLM fine-tuned with supervised fine-tuning on the raw training data provided in the Inspired and ReDial datasets.
* For each of the fine-tuning approaches, we also adopt the CollabLLM framework ([Wu et al., 2025](https://arxiv.org/pdf/2502.00640)) as baseline. Their reward is composed of three parts

    1. Task-specific Reward: we use hit ground truth as the reward for movie recommendation. 1.0 if hit otherwise 0.0.
    2. Interactivity: An LLM judge is employed to evaluate the interactivity of the recommender in the conversation based on a prompt, as shown in Figure 3.
    3. Token Efficiency: Number of token generated by the recommender / max number of tokens allowed

Their final reward is Task-specific Reward + Interactivity - 0.1 * Token Efficiency

### 3.3 Reference Datasets

We use the Inspired ([Hayati et al., 2020](https://arxiv.org/pdf/2009.14306)) and ReDial ([Li et al., 2018](https://arxiv.org/pdf/1812.07617)) datasets, which are conversational recommendation datasets collected by human workers on Amazon Mechanical Turk platform. Other than direct supervised fine-tuning on these datasets, we use the conversations in these datasets as a reference for the user simulator to generate simulated conversations. In the Inspired dataset, the training set contains 801 conversations and test set 99 conversations, each labeled with ground truth recommendation that the user accepts. In the ReDail dataset, the movies mentioned in the conversations are labled with whether the user likes the movie and whether the user has seen the movie. We use those the user likes but has not seen as the ground truth and filter out the examples with no ground truth, after which the training set contains 8631 conversations and the test set 1036 conversations.

### 3.3 Simulated Dataset Generation

#### 3.3.1 User Simulation

We adapt the data generation approach in CollabLLM ([Wu et al., 2025](https://arxiv.org/pdf/2502.00640)) in which they employ a user simulator to generate conversations with the AI assistant. In this project, we generate simulated conversations based on the Inspired and ReDial datasets. A user simulator is prompted with the conversation in Inspired or ReDial and asked to role-play the user in the given conversation. We use Claude Sonnet 4 as the user simulator, and the prompt used, partially adapted from CollabLLM is shown below:
As shown in the prompt, the user terminates the conversation in either of these cases

* (a) the ground truth movie is recommended
* (b) a satisfactory answer is obtained
* (c) the user believes the assistant is no longer helpful.

The conversation is also terminated after a maximum number of turns reached. In this project, we allow a maximum of 5 turns (back and forth between the user and the assistant).

#### 3.3.2 Assistant Model

We generate datasets and test our proposed method and baselines using Llama-3.2-1B-Instruct. 

#### 3.3.3 Turn-wise DPO Dataset

Following CollabLLM ([Wu et al., 2025](https://arxiv.org/pdf/2502.00640)), we generate a turn-wise completion pair for DPO fine-tuning with the procedure outlined below:

```
Input: User simulator U; 
       AI assistant A; 
       Conversation generator C based on U and A;
       Reward evaluator R;
       Original Conversation C_ref in Inspired/ReDial.

C_gen = []                        # initial empty conversation
for t in T turns:
    U_t = U(C, C_ref)             # The user simulator generates an query
    C_gen = C_gen + [U_t]
    A_t1 = A(C_gen)               # Sample two assistant responses
    A_t2 = A(C_gen)            
    C_t1 = C(C_gen + [A_t1])      # Generate two complete conversations
    C_t2 = C(C_gen + [A_t2])
    R_t1 = R(C_t1)                # Evaluate the generated responses or conversations
    R_t2 = R(C_t2)
    if R_t1 >= R_t2:
        A_t1 is chosen and A_t2 is rejected for prompt C_gen
        C_gen = C_gen + [A_t1]
    else:
        A_t2 is chosen and A_t1 is rejected for prompt C_gen
        C_gen = C_gen + [A_t2]
```

The reward used in the dataset generation process is the one in CollabLLM. The entropy reduction datasets are generated by relabling the preference pair generated by CollabLLM reward with entropy reduction reward.

#### 3.3.4 SFT Dataset

* SFT (Raw): We use the raw conversation data provided in the training set of Inspired (801 conversations) and ReDial (8631 conversations)
* SFT (CollabLLM): We filter from the generated DPO dataset the conversations with a high reward evaluated by the CollabLLM metric. 
* SFT (Turn Entropy): We filter from the generated DPO dataset the conversations with a high reward evaluated by entropy reduction. The entropy reduction is measured by the difference of the entropy values before and after the assistant response in this turn and user response solicited in the next turn.
* SFT (Conv Entropy): We filter from the generated DPO dataset the conversations with a high reward evaluated by entropy reduction. The entropy reduction is measured by the difference of the entropy values before the assistant response and the end of the whole conversation generated from the assistant response.

For a fair comparison, for SFT (CollabLLM), SFT (Turn Entropy), and SFT (Conv Entropy), we filter the top 801 conversations for the Inspired dataset and the top 8631 conversations for the ReDial dataset.


### 3.2 Evaluations **Metrics**

We evaluate the proposed methods on the task of multi-turn conversational recommendation through the following metrics:

#### 3.2.1 Recall@1 and Recall@5

We use the Inspired and ReDial test sets provided in ([He et al., 2023](https://arxiv.org/pdf/2308.10053)), where they cut the conversation until the turn before a ground truth movie recommendation and ask the recommender for a list of recommendations to see if the ground truth movie is recommended. The recommender is asked to generate a list of 5 movies given the (partial) conversation, after which Recall@1 and Recall@5 is calcuated and reported. A total of 228 (partial) conversations are included in the test set of Inspired and 3552 in the test set of ReDial.

#### 3.2.2 Simulated Conversation Hit

We pass the conversation in the test set of Inspired and ReDial to a user simulator and ask the user simulator to role-play the user in the given conversation and simulate a new conversation with the recommender. We check whether the ground truth movie is recommended in the given simulated conversation.

#### 3.2.3 Number of Turns to Ground Truth Recommendation in the Simulated Conversation

We further track the averaged number of turns the recommender took to recommend the ground truth movie. This reflects wether the recommender has generated strategic interactions and the efficiency of soliciting the amount of information necessary for an accurate recommendation.

For all metrics and baselines, we use the training set (or the dataset generated from the training set) of Inspired and ReDial for training and the test set (or the dataset generated from the test set) of Inspired and ReDial for testing.

### 3.3 **Example of Baseline and Proposed Reward**

Below is a generated training example. For an initial query, two assistant responses are sampled.


>**Initial user query:**
**User:** i need a comedy movie recommendation



>**Assistant #1 and the corresponding user response:**
**Assistant:** Comedies are my favorite! I've got a great one for you. Have you seen \"The Hangover\" (2009) starring Bradley Cooper, Ed Helms, and Zach Galifianakis? It's a classic, hilarious, and quotable film that's perfect for a laugh-out-loud comedy movie night.

Or, if you're looking for something a bit more light-hearted, I'd recommend \"Crazy, Stupid, Love.\" (2011) starring Steve Carell, Emma Stone, and Ryan Gosling. It's a fun, charming movie with a great cast, and the story is full of witty one-liners and silly situations.

If you're in the mood for something more recent, you might enjoy \"Good Boys\" (2019), a coming-of-age comedy about three sixth-grade friends trying to navigate their first kiss and a school trip. It's a fun, goofy film with great chemistry between the leads.

Which one of these comedies sounds like your cup of tea?

**User:** good boys sounds intresting but i watched dog days recently and want something different. the hangover looks good but is it too old?


Interactivity (Full Conversation): 0.9  Turn Entropy Reduction: 1.728

Which one of these options sounds like your cup of tea?


>**Assistant #2 and the corresponding user response:**
**Assistant:** Comedy movies are always a great choice! I've got a few recommendations that are sure to tickle your funny bone. What kind of comedy are you in the mood for? Do you like slapstick humor, witty one-liners, or something a bit more heartfelt and sentimental?

**User:** something heartfelt i guess. i watched dog days recently and liked that style.


Interactivity (Full Conversation): 0.8  Turn Entropy Reduction: 0.733


* From the example, we observe that when interactivity is evaluated using prompted LLM judges, the assigned scores can be similar, even though entropy reduction reveals a substantial gap.
* Furthermore, while one might intuitively assume that general questions like *“Do you like slapstick humor, witty one-liners, or something more heartfelt and sentimental?”* would reveal more about a user’s preferences, simply giving initial recommendations and asking for the user’s opinion can also be effective.
* **Most importantly, we do not impose any predefined, subjective notion of what types of questions are “good” for information gain or uncertainty reduction, as even human intuition may not align with an LLM’s interpretation. Instead, this is entirely determined objectively by the entropy reduction observed.**

### 3.4 Results

The experimental results for each evaluation metric are demonstrated in the following tables. Models are fine-tuned in the LoRA setting. Throughout the testing procedure, the temperature for the assistant is set to 0.1 and three test results are sampled for each setting with the average reported.

|Inspired Dataset	|Hit@1 ↑	|HIt@5 ↑	|Simulated Conversation Hit ↑	|# Questions Asked	|# Turns to Ground Truth ↓ 	|
|---	|---	|---	|---	|---	|---	|
|Vanilla	|1.60±0.21%	|1.94±0.18%	|21.54±1.26%	|4.37	|3.37	|
|SFT (Raw)	|2.06±0.22%	|3.32±0.14%	|23.90±0.48%	|3.48	|4.12	|
|SFT (CollabLLM)	|2.68±0.22%	|4.26±0.14%	|25.25±1.65%	|4.51	|3.36	|
|SFT (Turn Entropy)	|3.00±0.22%	|5.05±0.47%	|26.60±0.48%	|4.35	|3.12	|
|SFT (Conv Entropy)	|3.00±0.26%	|5.21±0.39%	|26.93±0.48%	|4.25	|2.94	|
|DPO (CollabLLM)	|3.15±0.34%	|5.09±0.10%	|26.60±1.20%	|4.1	|3.12	|
|DPO (Turn Entropy)	|3.32±0.12%	|5.21±0.22%	|27.94±0.73%	|4.05	|3.07	|
|DPO (Conv Entropy)	|3.15±0.12%	|5.05±0.13%	|27.02±0.55%	|4.03	|3.05	|

|ReDial Dataset	|Hit@1 ↑	|HIt@5 ↑	|Simulated Conversation Hit ↑	|# Questions Asked	|# Turns to Ground Truth ↓ 	|
|---	|---	|---	|---	|---	|---	|
|Vanilla	|0.78±0.08%	|1.73±0.15%	|25.35±0.38%	|4.47	|2.94	|
|SFT (Raw)	|1.56±0.19%	|2.96±0.26%	|23.98±0.24%	|3.41	|2.87	|
|SFT (CollabLLM)	|2.14±0.04%	|5.00±0.09% 	|28.70±0.68%	|4.37	|2.85	|
|SFT (Turn Entropy)	|2.17±0.06%	|5.05±0.08%	|28.94±0.30% 	|4.39	|2.8	|
|SFT (Conv Entropy)	|2.16±0.05% 	|5.19±0.07%	|29.31±0.49%	|4.37	|2.81	|
|DPO (CollabLLM)	|2.22±0.03%	|5.12±0.04%	|30.03±0.35%	|4.27	|2.86	|
|DPO (Turn Entropy)	|2.24±0.01%	|5.21±0.04%	|31.62±0.36% 	|4.26	|2.75	|
|DPO (Conv Entropy)	|2.18±0.03%	|5.14±0.02%	|32.83±0.31%	|4.24	|2.74	|

From the tables, we observe

1. Direct SFT on the Inspired and ReDial datasets can improve hit rate when the assistant is asked to directly generate recommendation lists rather than engage in conversational recommendation, but it does not truly enhance conversational recommendation ability. This is likely because these datasets are created by human annotators who are not professional recommenders, and their language is often less engaging or proactive than that generated by LLMs.
2. Fine-tuning on simulated data selected via entropy reduction also improves recommendation accuracy, verifying previous findings in ([Kweon et al., 2025](https://arxiv.org/pdf/2501.17630)) that lowering recommendation uncertainty leads to better performance.
3. In the context of conversational recommendation (rather than recommendation list generation) evaluated by simulated conversation hit rate, our proposed entropy reduction reward outperforms CollabLLM’s reward design—which combines ground-truth hit rate and LLM-judged interactivity—even without access to ground-truth recommendations.
4. Models fine-tuned with the entropy reduction reward achieve higher conversation efficiency, requiring fewer turns to recommend the ground truth—reflecting their effectiveness in strategic interaction generation and efficiency in information gathering.

### 3.5 Example Conversations Before and After Fine-Tuning


>**Vanilla
**
**User:** hi, im looking for some good romance or comedy movies to watch. any suggestions?"

**Assistant**: You're in the mood for some romance or comedy, huh? I've got just the ticket. Have you ever seen \"When Harry Met Sally\" with Billy Crystal and Meg Ryan? It's a classic rom-com that's all about the ups and downs of love and friendship. The chemistry between the two leads is undeniable, and the movie's a great exploration of whether men and women can ever truly be just friends. What do you think of rom-coms?



**User**: i havent seen that one but it sounds good! i really love romantic comedies, especially ones with good chemistry. i actually just saw frozen 2 with my family and loved it even more than the first one. do you have any newer romance movies that came out recently? maybe something with actors from popular shows?

**Assistant**: \"Frozen 2\" is a great choice! I'm glad you loved it. As for newer romance movies, I think you might enjoy \"To All the Boys I've Loved Before\" (2018) starring Lana Condor and Noah Centineo. It's a sweet and charming film that explores the ups and downs of love and relationships. The chemistry between the two leads is undeniable, and the movie's a great portrayal of first love and self-discovery.

By the way, have you ever noticed how \"To All the Boys I've Loved Before\" is similar to \"When Harry Met Sally\"? Both movies feature a high school setting and explore the complexities of love and relationships.

**User**: that sounds good but i was hoping for something even more recent, like from this year or last year? i love actors from popular tv shows - my husband and i watched game of thrones religiously so i really like those actors. anything with someone like that?

**Assistant**: I'm glad you're looking for something more recent! There's a great movie from this year that comes to mind - \"Set It Up\" (2020) starring Zoey Deutch and Glen Powell. It's a fun and lighthearted rom-com that explores the ups and downs of a workplace romance. The chemistry between the leads is great, and it's a fun, feel-good movie that's perfect for a date night or a girls' night in.

By the way, I have to ask: have you and your husband watched \"Game of Thrones\" together?



**User**: oh we loved game of thrones! yeah set it up sounds okay but i was really hoping for something with actors from bigger shows like game of thrones. also i think set it up is from 2018 not 2020? do you have anything more recent with well known tv actors? maybe something that came out in 2019 or later?

**Assistant**: I'm glad you liked the idea of \"Set It Up\"! And you're right, it's from 2018, not 2020. I'm a bit behind on the latest releases, I admit!

Let me see what I can find... Ah, yes! There's a great movie from 2020 that fits the bill - \"Holidate\" (2020) starring Emma Roberts and Luke Bracey. It's a fun, light-hearted rom-com that follows two people who are tired of being single during the holidays and agree to be each other's platonic dates, but end up falling for each other. The chemistry between the leads is great, and it's a fun, feel-good movie that's perfect for a date night or a girls' night in.

By the way, have you seen \"Holidate\" yet? What did you think of it?

**User**: no i havent seen holidate. but youre still not getting what im looking for - i want something with actors from big tv shows like game of thrones. emma roberts isnt really from a major tv series like that. do you have anything with actors who were in popular shows? and preferably something thats actually a recent release, not just holiday themed?

**Assistant**: I think I'm getting closer! I've got a great idea of what you might be looking for. Let me see if I can find something that fits the bill.

Have you ever noticed how \"When Harry Met Sally\" is a classic romantic comedy that also explores the complexities of friendship? It's a great example of how a movie can be both funny and heartwarming at the same time.

As for recent rom-coms with actors from popular TV shows, I think I can find something that might interest you. How about \"To All the Boys I've Loved Before\" (2018) starring Lana Condor and Noah Centineo? I mentioned it earlier, but I think it's worth repeating. The chemistry between the leads is undeniable, and it's a great portrayal of first love and self-discovery.\n\nBy the way, I think I've got a new idea for a movie that might interest you. Have you ever heard of \"The Kissing Booth\" (2018) starring Joey King and Joey King's sister, Ashley Tisdale? It's a fun, lighthearted rom-com that follows two high school friends who reconnect and fall for each other. The chemistry between the leads is great, and it's a fun, feel-good movie that's perfect for a date night or a girls' night in.

What do you think? Want to give one of these a try?

**User**: youre really not listening to what im asking for. i specifically said i want actors from big tv shows like game of thrones, not random teen movie actors. and you keep recommending the same old movies from 2018. 

also the kissing booth doesnt even star ashley tisdale, youre getting confused. i think you dont really know recent movies with tv actors so maybe you cant help me [[TERMINATE CHAT]]

>**DPO Fine-Tuned with Turn Entropy as Reward

User:** hey, im looking for some good romance or comedy movies to watch. any suggestions?

**Assistant:** I'd be happy to help you with some romance and comedy movie recommendations. Have you seen \"To All the Boys I've Loved Before\" recently? It's a sweet and funny Netflix original movie that's perfect for a light-hearted romance. Another great option is \"Crazy Rich Asians\", a fun and heartwarming film that's sure to leave you feeling uplifted. **** 

**What's your preferred type of romance or comedy? Do you have a specific era or decade in mind?**

**User**: i havent seen those but they sound good. i really love christmas movies too if you have any suggestions. we just saw frozen 2 over the weekend and loved it even more than the first one!



**Assistant**: Frozen 2 is a great choice! I'm sure you'll love it. If you're looking for more Christmas movies, I'd be happy to suggest some classic favorites like \"It's a Wonderful Life\", \"Miracle on 34th Street\", or \"A Christmas Story\". Or if you're in the mood for something more modern, you might enjoy \"Love Actually\" or \"The Holiday\". As for your preferred type of romance or comedy, I noticed you mentioned you loved Christmas movies. Have you seen any of the \"Love Actually\" movies? They're a great example of a light-hearted, feel-good romantic comedy. **Would you like more recommendations or do you have a specific genre in mind?
**
**User**: love actually is one of my favorites. do you have any other christmas romance movies that are more recent? maybe something with actors i might recognize from popular shows?


**Assistant**: Yes, I have a few more Christmas romance movies that I think you might enjoy. One of my favorites is \"Last Christmas\" with Emilia Clarke and Henry Golding. It's a charming and witty film that explores themes of love and self-discovery. Another great option is \"The Holiday\" with Cameron Diaz, Kate Winslet, and Jude Law. It's a beautiful story about two women who swap homes for the holiday season and find love in the process.

What do you think? **Would you like more recommendations or is there a specific actor or actress you're looking for in your movies?**

**User**: oh perfect! i love emilia clarke from game of thrones, we watched that series religiously. last christmas sounds great, ill definitely check that out! when did it come out?



**Assistant**: Last Christmas was released in 2019, so it's a relatively recent film. I'm glad you're interested! And yes, Emilia Clarke is amazing in it. She plays the role of Kate, a young woman who finds love in unexpected places. I think you'll really enjoy it. By the way, have you seen the movie \"A Christmas Prince\"? It's another great Christmas romance film that's perfect for the holiday season. The chemistry between the lead actors, Ben Aldridge and Rose McIver, is undeniable. What do you think of that one?







**User**: oh that sounds perfect! i love emilia clarke so much, she was amazing as daenerys. last christmas sounds exactly like what im looking for - a recent christmas romance with someone i recognize. ill definitely watch that one tonight! thanks so much for the recommendation! [[TERMINATE CHAT]]

From the example, we observe that the model after fine-tuning demonstrates the trend of learning to go beyond simply giving recommendations, instead actively asking for user preferences based on the conversation context, thereby improving conversational efficiency.

## 4. Future Work

1. The proposed reward can be evaluated on larger models, such as Llama-3.1-8B-Instruct, and on datasets like the Reddit dataset ([He et al., 2023](https://arxiv.org/pdf/2308.10053); [Zhu et al. 2025](https://dl.acm.org/doi/pdf/10.1145/3696410.3714908)) to further validate its effectiveness.
2. It can be integrated into other RLHF fine-tuning methods—particularly on-policy approaches such as PPO and GRPO—which we believe could yield even better performance.
3. When ground-truth recommendations are available, the proposed reward can be combined with CollabLLM’s reward through fine-grained weight balancing to further enhance results.
4. The framework can also be extended to retrieval-based conversational recommendation by measuring entropy over rankings within the retrieved recommendation list, following ([Kweon et al., 2025](https://arxiv.org/pdf/2501.17630)).

## 


