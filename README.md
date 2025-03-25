# deep-learning

One-Hot Encoding and Trainable Word Embeddings 


Overview -

One-Hot Encoding - 
• Creates a binary vector for each category where only one element is "hot" (1) and others are 0 
• Dimensionality equals the number of unique categories 
• No inherent relationship between categories 
• Fixed representation that doesn't change during training 

Trainable Word Embeddings - 
• Learns dense, low-dimensional vector representations during training 
• Dimensionality is a hyperparameter (typically much smaller than one-hot) 
• Captures relationships between categories through learned similarity 
• Representation improves as the model trains 


Performance Comparison -

Advantages of One-Hot Encoding - 

Simplicity: Easy to implement and understand 
No information loss: Preserves all categorical distinctions 
Works well with small categorical spaces 
Deterministic: Doesn't require learning 
Advantages of Trainable Word Embeddings - 
Dimensionality reduction: More efficient for large vocabularies 
Learned semantics: Can capture meaningful relationships 
Better generalization: Often improves model performance 
Memory efficiency: Requires less space for large categorical variables 


Observations from Text Generation Experiment -

1. Training Performance - 
Convergence Speed : 
Trainable embeddings achieved lower loss values consistently across all epochs (6.27 → 2.25 
vs 6.55 → 2.55) 
Faster convergence suggests embeddings learn more efficient representations 
Training Time : 
Embeddings were 4.5x faster (112s vs 510s) 
Significant computational advantage despite same epoch count 
Final Loss : 
Embeddings achieved better final loss (2.25 vs 2.55) 
12% relative improvement in optimization 

2. Generated Text Quality - 
One-Hot Output : 
Shows basic syntactic correctness but limited semantic coherence 
Repetitive structure ("i tell it, i know in the") 
Lacks contextual flow between phrases 
Embeddings Output : 
Displays richer poetic imagery ("sweep of easy wind and downy flake") 
Better lexical diversity ("plentiful manikins skipping around") 
More creative word combinations suggesting learned semantic relationships 

3. Technical Tradeoffs - 
Memory Efficiency : 
One-hot encoding with 4,752 vocabulary size creates 4,752-dim sparse vectors 
Embeddings likely used <<100 dimensions (typical range 50-300) 
Representation Power : 
One-hot treats all words as equally distant 
Embeddings naturally cluster semantically similar words 


Key Findings -

• For text generation, embeddings clearly outperform one-hot encoding in : 
Training efficiency (time and convergence) 
Output quality (coherence and creativity) 
Memory usage (despite larger raw vocabulary) 
• Surprising Result : 
The performance gap is larger than expected - embeddings are both faster AND better, contrary 
to the common assumption that better quality requires more computation 


Recommendations -

• Always prefer embeddings for NLP tasks with vocabulary sizes >1000 
• The observed 4.5x speedup justifies using embeddings even for prototyping 
• For production systems, the memory savings become increasingly valuable at scale 


Limitations -
 
• Without perplexity/metrics, qualitative assessment is subjective 
• Hardware differences could affect timing results 
• Small sample size of generated text (single example each)