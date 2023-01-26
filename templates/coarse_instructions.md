## Instructions for COARSE-grained evaluation on PubMed

Please read all instructions before starting the annotation.

### Setup

1. Start by signing up on Label Studio, you will need to provide an email ID and password. It’s okay to use a non-existent throw-away email ID here. Also, do not use any personal / sensitive passwords (but make sure to remember your email / password for logging in next time!). Click on the box saying “<your name> — Summarization Evaluation”

2. In this batch a total of 30 summaries need to be evaluated. Every three consecutive rows are different summaries of the same source document. You can evaluate a summary by clicking on a row, and annotating it. Optionally, you can click on “Label All Tasks” at the top of the screen.

### Annotation Task

Each summary needs to be evaluated for its “correctness”. You need to provide a 0-5 judgment for the entire summary, where “correctness” can be defined as, “The absence of factual errors in the summary, where a factual error is a statement that contradicts the source document, or is not directly stated, heavily implied, or logically entailed by the source document”. For example,

Source Document (snippet shown) = ….. Vitamin C was discovered in 1912, isolated in 1928, and, in 1933, was the first vitamin to be chemically produced. It is on the World Health Organization's List of Essential Medicines. Vitamin C is available as an inexpensive generic and over-the-counter medication. Partly for its discovery, Albert Szent-Györgyi and Walter Norman Haworth were awarded the 1937 Nobel Prizes in Physiology and Medicine and Chemistry, respectively. Foods containing vitamin C include citrus fruits, kiwifruit, guava, broccoli, Brussels sprouts, bell peppers, potatoes, and strawberries. Prolonged storage or cooking may reduce vitamin C content in foods. ….

Summary #1 (snippet shown) = … **Chicken contains vitamin C** …  
Summary #2 (snippet shown) = … Albert Szent-Györgyi won the **1955** Nobel Prize for discovering Vitamin C …  
Summary #3 (snippet shown) = … Vitamin C was the first chemically produced Vitamin …  
Summary #4 (snippet shown) = … **Apple contains vitamin C** …  

Errors marked in bold. Here, the snippets for summary #1 are incorrect, summary #2 partially correct, and summary #3 completely correct with respect to the source document. Summary #4 is incorrect with respect to the source document (since it’s never discussed), but a globally correct fact. You should treat such a summary as incorrect since it is not mentioned in the source document.

(This is an illustrative example only, the actual annotation task has much longer summaries / source documents.)

The rating scale is from 0 to 5, where 0 is the lowest possible rating (most or all of the summary is wrong / irrelevant to the source document), and 5 is the highest rating (most or all of the summary is correct).

While it is compulsory to provide a judgment from 0 to 5 for each summary, you can optionally provide additional comments in your annotation. For instance, if the judgment needs to be more nuanced than a 5-point scale, you prefer to mark something like “3.5”, or you would like to add some other notes about your judgment.

### Suggested workflow

Every three consecutive rows contain different summaries for the same source document. We suggest the following workflow while annotating documents —

1. Spend the first 10-15 minutes reading the source document and getting a general sense of the facts mentioned in the document.

2. Spend 5 minutes to read and annotate the summaries in each of the three consecutive rows which correspond to the same document. Add optional comments / notes if necessary.

3. In the last 5 minutes, recalibrate your ratings across the three rows if needed (for instance, you significantly preferred the correctness of summary 1 vs summary 2, but you gave it the same rating in the initial pass). Add optional comments / notes if necessary.
