# Evaluating my 2020 MLB Predictions: Part 1, Pete Alonso

## Partial Pooling For the Win

At the beginning of March I published some predictions about [how many home runs Pete Alonso would hit this season](https://dantegates.github.io/2020/03/03/predicting-pete-alonso's-2020-performance.html).

Rather than predicting a single number of home runs that we expected Alonso to hit, the predictions provided a range of how many home per number of plate appearances. In hindsight this was fortuitous because it allowed us to evaluate the predictions even after the MLB season had been sidelined for three and a half months before being truncated to a 60 game season (opposed to the normal 162 game schedule).

So how did the predictions fare?

The figure below shows that Alonso's actual production is contained within the confidence interval for the entirety of the season. This is the first thing I like to check when evaluating this sort of model. Outcomes that often jump outside of the confidence interval indicate a bad fit.

The next thing I like to consider is how the width of the intervals compare to the outcomes. We could easily satisfy the criteria above by producing very large confidence intervals, without actually providing information that's useful. What we want to see are intervals that contain most of the data within reasonable bounds.

What we see below is that the intervals seem to be just "wide enough", even accounting for Alsono's [early season slump](https://risingapple.com/2020/08/04/mets-pete-alonso-slump-2020/).

This is a really good sign that we got a decent fit. As described in my original post, one of the motivations behind this model was the challenge presented by Alonso only having one year of big league experience in which he hit the ground running and never slowed down (recall he won Rookie of the Year and set the record for most home runs hit in a rookie season). The fact that the model accounted for Alonso getting off to a slow start indicates that the model didn't overfit to his 2019 campaign - partial pooling for the win!

![](pete-alonso-2020-evaluations.png)

Despite these struggles and letting his stats such as batting average, on base percentage and slugging drop compared to last year, given his total appearances, our model had an 88% chance of him hitting less than the 16 home runs he managed to pull off in the end. Even though he's a division rival, my hat's off to Alonso for pulling out of his slump and getting back on track.