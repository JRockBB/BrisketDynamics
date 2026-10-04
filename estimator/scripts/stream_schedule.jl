# Arrival scheduling is separate from graph node placement. A busy estimator
# consumes the backlog present when it becomes available; it never waits for
# the next state node or includes a reading that arrives during this solve.
function stream_arrival_batch(recs, i::Int, available_s::Real; paced::Bool)
    start_s = paced ? max(recs[i].t, available_s) : recs[i].t
    j = i
    while j < length(recs) && recs[j + 1].t <= start_s
        j += 1
    end
    return j, Float64(start_s)
end

# An estimate stamped at t is only available after its computation finishes.
# Return nothing when no committed estimate was available by the score time.
stream_available_at(rows, t) = findlast(r -> r["committed"] && r["processing_finish_s"] <= t, rows)
