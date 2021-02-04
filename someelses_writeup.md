**Setup**:

* Use C-Lightning and [the official rebalancing plugin](https://github.com/lightningd/plugins)
* Good returns possible with *significantly less* than 0.30BTC and *significantly fewer* than 30 channels.
* Calculate collected fees with [this script](https://snippet.host/ncsw/raw).
* Calculate rebalancing fees paid with [this script](https://snippet.host/xdqh/raw).
* These strategies are specific to routing nodes.

**Pointers**:

* Large capacity channels: At least 800,000 satoshis, ideally 1,000,000+.
* Choose nodes that improve your *low-fee routing diversity*.
* Always have balanced incoming and outgoing capacity, +/- 5% or less.
* Use (mostly) static fees.
* Your fees should be higher than your channel peers, on average. This allows you to rebalance for cheaper than you charge for forwarding payments.
* For channels that seems highly biased to the outgoing direction, raise the forwarding fee above the cost of rebalancing that channel.
* **Always keep your channels balanced**, at least within 10%. 
* Dynamically rebalance unbalanced channels.
* Never pay a rebalancing fee higher than your outgoing fee rate on the channel being rebalanced towards you.
* Gain incoming capacity... somehow.
* Gain outgoing capacity. Create a new channel ~50,000 satoshis bigger than the amount of outgoing capacity you'd like to gain to an infinite liquidity sink like the LOOP or Boltz nodes. Set a very high fee rate on this channel (say, 2-5%) and you will soon have people paying you to help balance your liquidity.  Once all the capacity is drained to the other side, issue a mutual close during a low-fee period.




**Channel peer selection with "low-fee routing diversity":**

I have seen a number of metrics used to gauge which potential channel peers would be most beneficial. I don't think there is any one "correct" metric to use, but I have found success with mine: low-fee routing diversity. The idea is that we want to choose channel peers who offer us the greatest benefit to routing diversity across the LN below a given feerate threshold. Since during the payment process LN nodes try lower-fee routes before moving on to higher-fee routes, by increasing the number of ways nodes can reach you/be reached by you at low fee rates, you increase the probability that your node will lie along a successful payment route and thus your share of the LN payments market.

I have [a super crappy python script](https://snippet.host/apmk/raw) that I use to evaluate this metric. It takes super long to run (even when compiled with Cython), is not at all optimized, and is surely buggy (at the very least I believe there is a bug in the maxflow geomean calculation) but its channel proposals have worked out well.

Here's how to use it:

* Collect the LN channel graph at multiple times throughout the week: `lightning-cli listchannels >lnchannels.20210202`.  
* Collect periodically because a good channel peer at one time may not be a good channel peer at other times. This helps identify peers that are consistently good choices over time.
* For each collected LN channel graph, analyze the graph for varying fee rates, e.g.: `<lnchannels.20210202 ./lowfee_routing_diversity.py <your node pubkey> 2033 250`, `<lnchannels.20210202 ./lowfee_routing_diversity.py <your node pubkey> 1050 150`, `<lnchannels.20210202 ./lowfee_routing_diversity.py <your node pubkey> 1010 75`
* Note which prospective peers (identified by their pubkey) are consistently in the "top 10" nodes for varying fee rate thresholds and with varying snapshots of the LN channel graph. 
    * The reported "peer benefit" metric is synthetic, but bigger is better. 
    * "Low fee routability improvements" is the number of LN nodes to which you have more low-fee reachable paths at the given fee rate threshold, if you were to peer with that node. 
    * "Bonus" is a measure of reachability improvement to nodes with few existing low-fee paths. "maxflow geomean" is supposed to be the geomean of the maxflows to each LN node, calculated on a unit-capacity graph composed entirely of channels reachable below the threshold routing cost. I am not sure this last metric is correctly calculated, but higher is better.
* Open a channel to one or more of the most consistently high-rated proposed peers.


**Dynamically Rebalance Channels**
* Rebalancing channels is essentially undoing transactions en masse. 

Channels are defined as a directed edge with inflow and outflow. As one increases, the other decreases.
The sum of inflow and outflow (capacity) is constant. 
Transactions change the inflow/outflow of two channels identically. 
Channels can be balanced in pairs. Balancing a pair of channels is associated with a cost.
The goal is to minimize the cost to rebalance all channels, by picking a set of pairs to rebalance. 
Conjecture: This is easier if there is an even number of pairs.
This can be done with a depth first search. 

Channel balance can be reduced to one number, positive or negative depending on which side the greater balance is on 
Find pairs of imbalances whose sum of sums is minimal (assuming all rebalances cost the same)
