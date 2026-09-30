using Serialization,SHA

"Fingerprint a caller's complete graph state, including captured callable data and aliasing."
function target_state_fingerprint(graph)
    io=IOBuffer()
    serialize(io,graph)
    return bytes2hex(sha256(take!(io)))
end
