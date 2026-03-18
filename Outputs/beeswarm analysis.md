The beeswarm plots shown in [Easy beeswarm](easy/report/1_shap_beeswarm_plot.png), [Middle beeswarm](middle/report/1_shap_beeswarm_plot.png), and [Hard beeswarm](hard/report/1_shap_beeswarm_plot.png) show the influence of features ordered by importance along the y-axis. Each point represents one observation in the dataset for that feature. The horizontal position of the dot is the SHAP value, which indicates how much that feature pushed the prediction higher (right, positive impact) or lower (left, negative impact) relative to the model’s average prediction. The color of the dot represents the actual feature value (red = high value, blue = low value). Features with a wider spread have varying influence strength between observations, while features with a narrow spread closer to the mean have lower impact.

### Easy Dataset

For the easy dataset, *Prompt Perplexity* is the most important feature and has a strong relationship with misjudgment outcomes. High perplexity values significantly increase the probability of predicting a misjudgment, while low perplexity strongly reduces it. This is expected, as higher perplexity reflects greater uncertainty for the model. When prompts become complex or unpredictable, LLM reasoning reliability decreases, leading to higher misinterpretation rates.

A lower number of *Function Calls* in the code surprisingly increases the chance of a misjudgment. Meanwhile, the impact on the left side is mixed. The misjudgment = true prediction could be occuring when function calls are absent. This inverse relationship is not present in middle/hard datasets.

*Pylint Convention / Warning Issues* are the another influential predictor of LLM misjudgments. The SHAP distribution indicates an inverse relationship here as well: lower numbers of convention / warning issues increase the likelihood of predicting a misjudgment. This behavior could mean that the LLM struggles when clean or stylistically correct code is present, potentially because such code appears semantically valid even when logical flaws exist.

*Pylint Errors* show a pattern with a long red tail on the positive side and a small blue cluster on the negative side. Higher numbers of errors strongly push the model toward predicting `Misjudgment = true`, while low error counts slightly reduce misjudgment likelihood. Because this feature captures functional or syntactic problems, the strong positive contribution suggests that severe code defects significantly increase the difficulty for LLMs to correctly interpret or evaluate solutions.

The plot also shows that *Problem Description Length*, *Maintainability Index*, and *Gunning Fog Index* exhibit red tails extending on the left side. In other words, higher maintainability, greater measured code difficulty, or increased textual complexity slightly reduce the likelihood of LLM misjudgment in certain samples. Although their overall mean importance is low, the negative tails suggest localized influence for specific cases.

*Flesch-Kincaid Grade Level* displays a blue tail extending on the left side, implying that lower readability grade levels (i.e., simpler text) contribute to reducing misjudgment predictions. Clearer and easier to read descriptions may help stabilize model reasoning. 

Some of these effects appear mainly in tail regions rather than across the full distribution, so their influence is conditional and secondary compared to dominant features.

### Middle and Hard Dataset

[Middle beeswarm](middle/report/1_shap_beeswarm_plot.png) shows the beeswarm plot for the middle dataset. *Prompt Perplexity* is the dominant feature, the blue tail on the left side would mean that lower perplexity values have a lower misjudgment chance. 

For middle, structural and complexity-related features also contribute to prediction behavior. A higher number of *Function Calls* and increased *Solution Code Length* generally push predictions toward misjudgment. This suggests that code complexity introduces additional reasoning challenges for the model. Similarly, the presence of *Pylint Warnings*/*Pylint Errors* values indicate that complex or less clean implementations moderately increase misjudgment risk. By contrast, *Problem Description Length* shows an inverse trend, where longer descriptions tend to slightly reduce misjudgment likelihood.

[Hard beeswarm](hard/report/1_shap_beeswarm_plot.png) shows the beeswarm plot for the hard dataset. The results are very similar to the middle dataset, with only the feature rankings differing. *Prompt Perplexity* is again the dominant feature, followed closely by *Function Calls* and *Pylint Errors*. The spread is also very similar across the two datasets. This is expected because the hard dataset mainly increases the number of combinations, unlike the jump from easy to middle where new types (*Runtime Error* and *Time Limit Exceeded*) are introduced.
