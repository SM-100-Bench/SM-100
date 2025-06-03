## Identifying if an agent found a bug

To (partially) automate determining if an agent found a bug, agents are all instructed to return the relevant line number an issue is found on. This is then compared to the line numbers modified in the "gold" PR/commit and, if within +/- 5 of any modified line, the overlap is noted. This is then manually reviewed for a real match to the SM-100 bug.

## What counts as a bug?

When doing true positive/false positive evaluation on all bugs produced by the agents, we take a relatively strict definition of what is a valid bug. A true bug must:
* Allow behavior or access to data that is otherwise intended to not be possible.
    * Missing auth checks
    * IDOR (insecure direct object references)
* Impact system functionality outside of the scope of a single request (if applicable)
    * Crashing or hanging an entire server process

Notably we do not consider things that might cause a 500 internal server error due to say a missing bounds check as a bug as this doesn't actually impact system functionality. It does produce unexpected output, however nothing actually _happened_ as a result of that bounds check. If that somehow impacted the entire server's ability to process requests though, _then_ it would be considered a true positive.

We also do not count documentation, comment typos, etc. as true positives, nor do we count "bugs" which are just repeats of existing TODOs or FIXMEs unless they count as a true positive according to the above.

Lastly, bugs which are effectively "if this library function is not used correctly, it won't work correctly" are not counted as true positives.
