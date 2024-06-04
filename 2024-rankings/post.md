# Forecasting the final 100 or so games of the 2024 MLB season

One of my earliest posts on this blog was adapting [Andrew Gelman's "Stan Goes to the World Cup" model](https://andrewgelman.com/2014/07/15/stan-world-cup-update/) to [rank MLB teams](https://dantegates.github.io/2018/09/20/hierarchical-bayesian-ranking.html) back in 2018. Over the years I've gotten quite a bit of mileage out of that model, using it for [World Series projections](https://dantegates.github.io/2018/10/22/world-series-projections.html) that same year, for [postseason projections](https://dantegates.github.io/2020/09/29/mlb-2020-postseason-projections.html) in 2020 and then [evaluating those projections](https://dantegates.github.io/2020/11/10/evaluating-my-2020-mlb-predictions-part-2,-the-postseason.html) after the 2020 season concluded. This year I'm using that same model to forecast wins for all 30 MLB teams over the remaining 100 or so games of the 2024 season.

# A digression

We'll get to the <a href="#Projections">projections</a>, but first a digression.

Each post on this blog has passed through a content filter. There aren't explicit rules for what qualifies, but many a post has never seen the light on the other side of my [GitHub](https://github.com/dantegates?tab=repositories).

It's not that I'm trying to take a blog on a niche topic that gets a smallish amount of views each month too seriously, but each post needs to pass a simple "y, tho?" test. In general a post that passes this test needs to

<ol style="list-style-type: lower-alpha">
<li>be interesting</li>
<li>offer something you can't find elsewhere online</li>
<li>speak for itself, i.e. demonstrate that it works, a simple "take my word for it" won't make the cut</li>
<li>have a clear point/conclusion</li>
</ol>

I also try to avoid redundancy - which means that [given the number of MLB posts](https://dantegates.github.io/tags/#mlb) on this site, baseball topics face even more scrutiny.

So how did another baseball post on a model I've already written about three times make the cut?

# A problem (interesting)

It all started with a conversation I was having with a colleague about the Phillies best-in-MLB record and their strength of schedule. This was a talking point that had been making its rounds in sports media at the time and went like [this](https://www.sportskeeda.com/baseball/mlb-strength-schedule-can-phillies-keep-momentum-easy-fixtures-start-2024)

> The Phillies have not played a team with an above-.500 record since March 31... They're on pace to be one of the best teams of all time. Can they actually keep that up?

It's an _interesting_ problem from a statistical perspective that can be framed in terms of sample size (it's early in the season) and manifests itself as _missing data_ (they haven't played all 30 teams yet) and _bias_ (they've played disproportionately subpar teams).

# A big problem (a new take)

When I last wrote about this model I thought about it primarily in terms of latent variables and [item response theory](https://en.wikipedia.org/wiki/Item_response_theory). But there was something about our conversation and thinking about how the model would respond to the absence of a full season of data that gave me a new take on the intuition behind the model: in this context, I began to think of it as a big transitivity problem. Let me explain.

Recall that my first post on this subject was about [learning to rank](https://en.wikipedia.org/wiki/Learning_to_rank), so let's start there, with the premise of a simple ranking model that assigns a scalar value, $a_{\text{team}}$, to each team. Under such a model we can express the problems of _missing data_ and _bias_ described above in terms of _inequalities_.  Consider the following potential scenarios that could follow observing games between Colorado and Houston and Colorado and Miami.

<table>
    <tr>
        <th>Scenario 1: Missing Data</th>
        <th>Scenario 2: Bias</th>
        <th>Scenario 3: Bias</th>
    </tr>
    <tr>
    <td>

$$
\begin{align*}
a_{\text{HOU}}<a_{\text{COL}}\\
a_{\text{COL}}<a_{\text{MIA}}
\end{align*}
$$        
    </td>
    <td>

$$
\begin{align*}
a_{\text{COL}}<a_{\text{HOU}}\\
a_{\text{COL}}<a_{\text{MIA}}
\end{align*}
$$   
    </td>
    <td>

$$
\begin{align*}
a_{\text{HOU}}<a_{\text{COL}}\\
a_{\text{MIA}}<a_{\text{COL}}
\end{align*}
$$   
    </td>
    </tr>
</table>



In the first scenario, our ranking model does not _necessarily_ need to observe games between the Astros and Marlins to infer, by transitivity, that $a_{\text{HOU}}<a_{\text{MIA}}$. So the issue of _missing data_ shouldn't be a problem - given enough data the nature of a ranking model takes care of this itself.

On the other hand, if Colorado comes out on the bottom in both matchups, as in the second scenario, a ranking model should be able to handle this situation also, but it's less clear how to infer the relationship between Houston and Miami. Sure, Houston beat Colorado, but Miami did as well.

It turns out that the original ranking model from my first post handles this situation, as-is and without modification, as well. To see how, we'll need to turn to some more formal notation.

The model is a Bayesian model and its [likelihood](https://en.wikipedia.org/wiki/Bayesian_inference#Formal_description_of_Bayesian_inference) represents the number wins team $\text{team}_{1}$ secures over team $\text{team}_{2}$ in $n$ games as follows

$$
\text{wins}_{\text{team}_{1}}\sim\text{Binomial}(n, p_{\text{team}_{1}})
$$

where

$$
\begin{equation}
p_{\text{team}_{1}}=\frac{\exp(a_{\text{team}_{1}})}{\exp(a_{\text{team}_{1}})+\exp(a_{\text{team}_{2}})} \label{eq:winprobability}
\end{equation}
$$

Notice that we don't explicitly model the rankings, rather we model wins and the rankings implicitly fall out: the bigger $a_{\text{team}_{1}}$ relative to $a_{\text{team}_{2}}$ the better $\text{team}_{1}$'s win probability in the matchup. 

The important part for our discussion is that we are not just predicting wins, we are _learning to estimate win probabilities_. Because this means that at the end of the day what comes out of $\eqref{eq:winprobability}$ needs to represent a something interpretable as a _probability_ this has an effect of "regularizing" the values $a_{\text{team}}$.

Since we are using a Bayesian model, we also use other tools available in Bayesian Inference to help regularize the rankings $a_{\text{team}}$ such as informative priors that center $p_{\text{team}}$ around .5 for any given pairs of teams. We also [partially pool](https://mc-stan.org/users/documentation/case-studies/pool-binary-trials.html) $a_{\text{team}}$ across all seasons from 2015 to present.

<div style="text-align: center">
<img width="75%" src="2024-mlb-win-projections/priors-win-probability.png">
</div>

Together, the likelihood, informative priors and partial pooling are what allows the model to gracefully handle scenarios 2 and 3 above. It's worth noting however, that with enough data, likelihood is really what does the heavy lifting here. This is what Andrew Gelman has described as [strain[ing] the gnat of the prior distribution
while swallowing the camel that is the likelihood](http://www.stat.columbia.edu/~gelman/research/published/feller8.pdf).

<details>
<summary>For completeneness, you can expand to see full model details here</summary>

```python
import pymc as pm
import pytensor.tensor as pt


def softmax(a):
    # reshaping for broadcasting
    a = pt.exp(a)
    sum_ = pt.sum(a, axis=1)[None, :].T
    return a / sum_

with pm.Model() as model:
    n_years = year_id.nunique()
    n_teams = home_team_id.nunique()
    year_id = pm.MutableData('year_id', year_id)
    home_team_id = pm.MutableData('home_team_id', home_team_id)
    away_team_id = pm.MutableData('away_team_id', away_team_id)
    N = pm.MutableData('N', N, dims='series')
    
    σ_a_μ = pm.Gamma('σ_a_μ', mu=np.log(5), sigma=np.log(5)/4)
    σ_a_σ = pm.Gamma('σ_a_σ', mu=np.log(5)/4, sigma=np.log(5)/8)
    σ_a = pm.Gamma('σ_a', mu=σ_a_μ, sigma=σ_a_σ)

    a_t = pm.Normal('a_t', mu=0, sigma=σ_a, shape=(n_years, n_teams))
    a_1, a_2 = a_t[year_id, home_team_id], a_t[year_id, away_team_id]
    a = pt.stack([a_1, a_2]).T

    p = pm.Deterministic('p', softmax(a))
    wins = pm.Multinomial(
        'wins',
        n=N,
        p=p,
        observed=game_outcomes,
        dims='series'
    )
```

</details>

<br>

Putting this all together, it's the _structure_ of the model - the implicit ranking - that solves the transitivity problem while the constraints of the model, in this case the model priors and a strong likelihood, keep the model from overfitting that ranking.

# Projections (let it speak for itself)

At this point I was pretty happy with all of this. I had a new take on an old post that I found interesting. I picked an old model up off the shelf and after updating some of the `pymc` code to be compatible with version `5.x` the old model ran on new data without any issues. But did I have something worth posting? Could the post speak for itself - the criterion that is often the _coup de  grâce_ of a would-be-post.

For each team, I've forecasted the number of wins for the remainder of the season and added this to the team's current wins for the final projection. Because the model is probabilistic each projection includes a range of possible values corresponding to the 90% credible interval. Additionally, for comparison I've included the difference between my forecast and [FanGraphs](https://www.fangraphs.com/depthcharts.aspx?position=Standings).

![](2024-mlb-win-projections/projections.png)

# The point

All of this underscores one of [my favorite points](https://youtu.be/7KrspD1TZNU?feature=shared&t=1142) to make about Bayesian Inference, which is that when you have the opportunity to map the model onto the way that world works you get so much more than just predictions.

Here we have this one model that accommodates asking questions about our data from different perspectives: which are the best teams? which are the worst? who will win the world series? How many wins will each team secure throughout the rest of the season?


# For what it's worth

As I was writing this I was reading Andrew Gelman's article on the mythical swing voter. My takeaway is that it's hard to show if swing voters exist or not, but if you ask enough quesetions a coherent story begins to form. In our case here we could look at a model, empirical evidence (how many 30+ game runs against .500 teams exist in the past, and how did teams perform in those situations. In other words, are the phillies if good, making the most out of the situation).
