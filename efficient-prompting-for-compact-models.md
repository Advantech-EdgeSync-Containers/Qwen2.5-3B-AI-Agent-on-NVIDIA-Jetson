# Prompt Guidelines for Small & Quantized Models

These guidelines help you craft better prompts for small and quantized language models, which have limited reasoning and memory capabilities. Small and quantized models do best with clear, scoped, and well-formatted inputs. When in doubt, simplify the task and make your prompt explicit.

| S.No. | Guideline                                 | Description |
|-------|--------------------------------------------|-------------|
| 1 | Be Specific and Unambiguous | As these models have limited reasoning depth, they perform better when tasks are clearly stated. Avoid open-ended or vague language. <br><br>❌ Bad Prompt: "Tell me about physics."<br>✅ Good Prompt: "Explain Newton's first law of motion with an example." |
| 2 | Use Simplified Language | Avoid nested clauses, abstract phrasing, or metaphor-heavy language. <br><br>❌ Bad Prompt: “Expound on the sociopolitical ramifications of early capitalism.”<br>✅ Good Prompt: “What are the effects of capitalism on society?” |
| 3 | Ask Fact-Based or Procedural Questions | Avoid abstract or scattered questions. <br><br>❌ Bad Prompt: "Why is life meaningful?"<br>✅ Good Prompt: “What are the steps to change a car tire?” |
| 4 | Avoid Multi-Part Prompts | Stick to one idea per prompt. <br><br>❌ Bad Prompt: “What is gravity, how does it work, and how is it measured?”<br>✅ Good Prompt: “What is gravity?” |
| 5 | Use Formatting Cues When Appropriate | Formatting helps models structure output better. <br><br>❌ Bad Prompt: “What are some ways to stay healthy?”<br>✅ Good Prompt: “List 3 ways to stay healthy in bullet points.” |
| 6 | Avoid Time-Sensitive Facts | Models have a training cutoff and might not be up to date. <br><br>❌ Bad Prompt: “Who won the 2024 US presidential election?”<br>✅ Good Prompt: “How is the US president elected?” |
| 7 | Avoid Sarcasm, Irony, or Humor | Figurative language may be misunderstood. <br><br>❌ Bad Prompt: “Oh sure, because nothing says fun like quantum physics!”<br>✅ Good Prompt: "List 3 main features of quantum physics.” |
| 8 | Avoid Domain-Heavy Jargon Unless Explained | Small models may lack specialized vocabularies. <br><br>❌ Bad Prompt: “Explain the glyoxylate cycle in plants.”<br>✅ Good Prompt: “What is the role of special metabolic cycles in plants?” |
| 9 | Don’t Rely on Memory of Named Entities | Use descriptive references instead of assuming memory. <br><br>❌ Bad Prompt: “What did Turing say in the 1950 paper?”<br>✅ Good Prompt: “What was Alan Turing’s contribution to AI?” |
| 10 | Avoid Chaining Tasks (e.g., Generate + Evaluate) | Compound tasks are harder for small models. <br><br>❌ Bad Prompt: “Write a poem and then explain its meaning.”<br>✅ Good Prompt: “Write a short 2-line poem about rain.” |
| 11 | Use Concrete Nouns and Verbs | Abstract language or ambiguous pronouns are confusing. <br><br>❌ Bad Prompt: “They said it was inevitable.”<br>✅ Good Prompt: “Scientists said climate change is inevitable.” |
| 12 | Ask for Small, Atomic Code Snippets | Scope tasks minimally for better performance. <br><br>❌ Bad Prompt: “Build a complete blog app in Django with login, comment, and admin features.”<br>✅ Good Prompt: “Write a Django model for a blog post with title and content fields.” |
| 13 | Specify the Programming Language | Always mention the language. <br><br>❌ Bad Prompt: “How do I reverse a string?”<br>✅ Good Prompt: “How do I reverse a string in Python?” |
| 14 | Avoid Combining Explanation and Code Generation | Ask one thing at a time. <br><br>❌ Bad Prompt: “Explain recursion and write a recursive function to compute factorial.”<br>✅ Good Prompt: “Write a recursive Python function to compute factorial.” |