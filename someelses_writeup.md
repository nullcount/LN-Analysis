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
* Principle: Rebalancing channels is essentially undoing transactions en masse. 

Channels are defined as an edge with inflow and outflow. As one increases, the other decreases.
The sum of inflow and outflow (capacity) is constant. 
Transactions change the inflow/outflow of two channels identically. 
Channels can be balanced in pairs. Balancing a pair of channels is associated with a cost.
Minimize the cost to rebalance all channels, by picking a set of pairs to rebalance. 

Conjecture: This task is easier if there is an even number of pairs.

Assumptions: 
* The capacity of every channel is the same. 
* The cost to rebalance any pair of channels is the same.

Then, channel balance can be reduced to one number, positive or negative depending on which side the greater balance is on.

Separate positive and negative balances into two different groups. 
The goal is to find pairs of imbalances between these groups that lead to all channels being balanced within 10%.

ex:
* pos: [1,2,3]
* neg: [-1,-2,-3]
* pairs to rebalance: (-1,1),(-2,2),(-3,3)

Conjecture: There always exists at least one mapping to perfectly rebalance all of the channels.

If every value in each group is unique and the size of the groups are the same, there is exactly one way to perfectly rebalance all of the channels. 
Note: Perfect rebalances are cool, but within a 10% threshold, there may be a cheaper way to rebalance.

ex:
pos: [1,2,2,3]
neg: [-4,-4]

pairs to rebalance: 
(-4,1),


This example shows that the process of finding pairs has to be iterative. 


Algorithm:

```
sort the positive list in ascending order and the negative list in descending order
while the lists are not empty:
    pair symmetric imbalances and remove those items from the list. 
    pop the first element from each list, add them together, 
    If the sum is positive, append the sum to the positive list, maintaining order
    If the sum is negative, append the sum to the negative list, maintaining order
```
[Think of it like an order book for a stock :P](https://en.m.wikipedia.org/wiki/File:Order_book_depth_chart.gif)

(imbalances map to nodes, modify this algo to produce pairs of nodes) 

