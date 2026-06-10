# Can a reward model hear the rhythm of good writing?

LLMs leapt ahead at math and code because those have verifiable rewards. Writing has none, and using an LLM as a judge tends to reward bland, generic "slop." So we asked a simple question: does prose carry a rhythm, the cadence you hear while reading silently, that signals quality and that word-level models miss? It looked cheap to test and potentially valuable, so we tried it.

We first extracted prosody features (stress, syllables, phrase boundaries) and checked whether they separate great writing from AI slop. They did, weakly, and partly independent of vocabulary. That left a question we kept circling back to: is the signal genuinely weak, or were we just measuring it crudely?

So we kept probing. Order-aware features added nothing. Real reading data (EEG and eye-tracking) backed it only weakly. As an AI-text detector it held against weak models but collapsed against GPT-4o. A speech-grounded extractor barely helped. Each negative result sharpened the real question.

The decisive test: does prosody add anything beyond a strong judge, which is what a real reward model would use? We built a blind frontier LLM judge and compared. The judge separated quality almost perfectly, and prosody added nothing on top. A control even showed that what prosody still carried was era and register, not quality.

Our takeaway: prosody is real but redundant for writing quality once a capable judge is present. That is a clean negative, and clean negatives count as results. The lasting value was the rigorous, self-skeptical method, not the signal. Next, we run discourse coherence through the same gauntlet.
