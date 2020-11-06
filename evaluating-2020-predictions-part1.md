# Evaluating my 2020 MLB Predictions: Part  1, The Postseason

This was my third year in a row making some sort of world series predictions (you can find previous year's predictions [here](https://dantegates.github.io/tags/#mlb)). This time around, however, I took it one step further and predicted the outcomes of the entire MLB postseason.

This year I made two sets of predictions for the 2020 MLB season. Now that the season has completed and the dust has settled let's see how they did.

## Background

I'll briefly cover the minimum background necessary for those unfamiliar with major league baseball, or my [earlier post](https://dantegates.github.io/2020/09/29/mlb-2020-postseason-projections.html) which described my postseason model.

This year, 16 teams made major league baseball's postseason. The playoff format consisted of a bracket with four rounds named the  Wild Card Series, Division Series, Championship Series and the World Series. The completed bracket looks like this.

![](https://img.mlbstatic.com/mlb-images/image/private/w_2400,q_85,f_jpg/mlb/l5nasohsm9ahbmrcqvf8)

Rather than predict the postseason outcome directly (the equivalent of filling out a march madness bracket) I generated probabilities that team $X$ would make it at least  as far as round $Y$. In my post I published these predictions alongside forecasts from [fivethirtyeight](https://projects.fivethirtyeight.com/2020-mlb-predictions/) and [mlb.com](https://www.mlb.com/news/2020-mlb-postseason-predictions).

## Evaluations

Since our evaluations are probabilistic, we can look at the likelihood each forecast assigned to the actual outcomes, for example: "What was the model's likelihood of the Dodgers winning the World Series?" or "What was the model's likelihood that the Yankees would win the Wild Card, but advance no further?"

The average likelihood for each forecast looks like

|       Forecast          |   Average Likelihood |
|:----------------|---------------------:|
| DG              |             0.21445  |
| FiveThirtyEight |             0.24     |
| mlb.com         |             0.40625  |
| Random Chance   |             0.335938 |

For reference we've included a "Random Chance" model that assumes the outcome of every game is random, e.g. each team has a $1/16$ chance of winning the World Series, $1/8$ of winning the Championship Series, etc.

Interestingly, only mlb.com outperforms the Random Chance model on average which is further illustrated by considering each prediction individually. (Note that we aren't including teams that lost in the first round, since those are implicitly accounted for in the other predictions.)

![](2020-evaluations/forecasted-likelihoods.png)

Another interesting view is to compare how much more likely each outcome is according to the forecast than random chance alone  would suggest.

![](2020-evaluations/forecasts-vs-random.png)

## Taking a step back

So what do we make of these evaluations? This is  a machine learning  blog, shouldn't the story line be "data science for the win, experts lose?"

Perhaps, but perhaps not. It's important to take a step back and consider the assumptions behind each model.

Both FiveThirtyEight and myself produced probabilistic models based on many simulations (I ran 10,000 simulations, FiveThirtyEight ran 100,000) to determine the likelihood of the many ways the postseason could work out (Remember there are over 32,000 ways the season could actually work out). Thus our model is truly trying to explore the likelihood of all possibilities, and not just  optimizing for the most likely outcomes.

On the other hand, mlb.com's 12 analysts each took their best guess as to what they thought the most likely outcome to be. I averaged all 12 brackets to get probabilities for the convenience of including them alongside my own, but their analysts  were fundamentally trying to answer a different  question.

This helps explain why the aggregate mlb.com predictions place much larger probability on the Dodgers and Rays facing  each other in the World Series, with the Dodgers coming out on  top. If you made predictions based on regular season performance alone, this is exactly what you would expect.

Another metric we could consider is the total likelihood of the postseason outcome as a whole - the fundamental question our simulations addressed. That is

$$
\begin{align}
    & \ \text{Pr(Dodgers won the World Series)} \\
    \times & \ \text{Pr(Rays won the ALCS, lost the World Series)} \\
    \times & \ \text{Pr(Braves won the NLDS, lost the NLCS)} \\
    \times & \ \ldots
\end{align}
$$

This time find FiveThirtyEight assigning the greatest likelihood to the actual outcome, but still less than random chance alone.

|                 |   Total Likelihood |
|:----------------|-------------------:|
| DG              |        2.26652e-06 |
| FiveThirtyEight |        5.77997e-06 |
| mlb.com         |        0           |
| Random Chance   |        3.05176e-05 |

Not only that, but mlb.com gave a 0 percent chance to the actual postseason outcome. Does this mean data science can declare victory? Well, not really. Again mlb.com was answering a different question. It would be extremely unlikely for at least one analyst to predict the postseason bracket perfectly (the chances are 12 / 32,768).

## Conclusion

Okay, wait... did we just go on a digression only to show another 

## MLE is awkward (sometimes)

## Probability 101

However, neither myself nor FiveThirtyEight published predictions in this fashion. Rather we predicted the probability that a team would make it at least as far as round $X$. More specifically, when we said there was a 30% chance of a team winning the Division Series and possibly additional rounds. Therefore it was necessary to calculate the probability of, say the Braves winning the NLDS, as $\text{Pr(Braves win NLDS)}\times\text{Pr(Braves do not win the NLCS)}$, where the latter term can be calculated as $\prod_{t\in T}^{}{\text{Team } t \text{ makes it at least as far as the NLCS}}$ where $T$ is the set of all teams the Braves could have faced in the NLCS - For example $T$ does not include any National League teams that would have necessarily had to be eliminated had the braves won the NLDS.
