# SaddleSearch driver. stdout is the harness protocol, so the solvers stay quiet.
# One job is one run! of StaticDimer, BBDimer, SuperlinearDimer, or ODEDimer.
# The energy and the gradient come back from the harness. Translation is held
# at zero when the job is a lowest-mode solve.

using LinearAlgebra
using SaddleSearch

struct CapReached <: Exception end

mutable struct State
    calls::Int
    asks::Int
    cap::Bool
    x::Vector{Float64}
    v::Vector{Float64}
end

function hexfloat(value::Float64)
    return string(reinterpret(UInt64, value), base=16, pad=16)
end

function unhex(text::AbstractString)
    return reinterpret(Float64, parse(UInt64, text, base=16))
end

function read_vec()
    count = parse(Int, readline())
    if count == 0
        return Float64[]
    end
    parts = split(strip(readline()))
    if length(parts) != count
        error("vector announced $count values and sent $(length(parts))")
    end
    return Float64[unhex(String(part)) for part in parts]
end

function ask(state::State, point)
    coords = collect(Float64, point)
    println("eval")
    println(length(coords))
    println(join(hexfloat.(coords), " "))
    flush(stdout)
    status = readline()
    if status == "cap"
        state.cap = true
        throw(CapReached())
    end
    if status != "ok"
        detail = readline()
        error("potential $status $detail")
    end
    state.calls = parse(Int, readline())
    energy = unhex(readline())
    width = parse(Int, readline())
    parts = split(strip(readline()))
    if length(parts) != width
        error("gradient announced $width values and sent $(length(parts))")
    end
    gradient = Float64[unhex(String(part)) for part in parts]
    state.asks += 1
    return energy, gradient
end

function log_last(log, key)
    if !haskey(log.D, key)
        return nothing
    end
    values = log.D[key]
    if isempty(values)
        return nothing
    end
    return values[end]
end

function library_count(log)
    value = log_last(log, :numdE)
    if value === nothing
        return 0
    end
    return Int(value)
end

function build_method(fields, tol_trans, tol_rot, budget, translate::Bool)
    method = fields["method"]
    len = parse(Float64, fields["len"])
    verbose = 0
    if method == "static"
        step = translate ? parse(Float64, fields["a_trans"]) : 0.0
        return StaticDimer(;
            a_trans=step,
            a_rot=parse(Float64, fields["a_rot"]),
            len=len,
            tol_trans=tol_trans,
            tol_rot=tol_rot,
            maxnumdE=budget,
            verbose=verbose,
        )
    elseif method == "bb"
        step = translate ? parse(Float64, fields["a0_trans"]) : 0.0
        return BBDimer(;
            a0_trans=step,
            a0_rot=parse(Float64, fields["a0_rot"]),
            len=len,
            tol_trans=tol_trans,
            tol_rot=tol_rot,
            maxnumdE=budget,
            verbose=verbose,
            ls=StaticLineSearch(),
        )
    elseif method == "superlinear"
        step = translate ? parse(Float64, fields["maximum_translation"]) : 0.0
        return SuperlinearDimer(;
            maximum_translation=step,
            max_num_rot=parse(Int, fields["max_num_rot"]),
            trial_angle=parse(Float64, fields["trial_angle"]),
            trial_trans_step=parse(Float64, fields["trial_trans_step"]),
            extrapolate=parse(Int, fields["extrapolate"]) != 0,
            translation_method=get(fields, "translation_method", "CG"),
            len=len,
            tol_trans=tol_trans,
            tol_rot=tol_rot,
            maxnumdE=budget,
            verbose=verbose,
        )
    elseif method == "ode"
        return ODEDimer(;
            len=len,
            tol_trans=tol_trans,
            tol_rot=tol_rot,
            maxnumdE=budget,
            verbose=verbose,
        )
    end
    error("unknown method $method")
end

function iteration_budget(cap)
    # One run! keeps the BB step and the CG direction. The harness cap
    # aborts a search that does not stop; maxnumdE only covers the case
    # where the library counts a request the cache did not.
    return max(cap * 8, 8)
end

function normalised(vector::Vector{Float64})
    scale = norm(vector)
    if !isfinite(scale) || scale < 1e-14
        out = zeros(length(vector))
        out[1] = 1.0
        return out
    end
    return vector / scale
end

function write_done(status, message, state, energy, gradient, force_norm, overlap, curvature, library, steps)
    flat = replace(message, "\n" => " ")
    println("done")
    println("status $status")
    println("message $flat")
    println("calls $(state.calls)")
    println("library_numdE $library")
    println("steps $steps")
    println("energy $(hexfloat(energy))")
    println("force_norm $(hexfloat(force_norm))")
    println("overlap $(hexfloat(overlap))")
    println("curvature $(hexfloat(curvature))")
    println("x")
    println(length(state.x))
    if !isempty(state.x)
        println(join(hexfloat.(state.x), " "))
    end
    println("v")
    println(length(state.v))
    if !isempty(state.v)
        println(join(hexfloat.(state.v), " "))
    end
    println("gradient")
    println(length(gradient))
    if !isempty(gradient)
        println(join(hexfloat.(gradient), " "))
    end
    println("end")
    flush(stdout)
end

function run_job(fields, x0, v0, ref)
    task = fields["task"]
    translate = task != "mode"
    force_tol = parse(Float64, fields["force_tol"])
    overlap_target = parse(Float64, get(fields, "overlap", "0.99"))
    cap = parse(Int, fields["cap"])
    len = parse(Float64, fields["len"])
    tol_rot = parse(Float64, fields["tol_rot"])
    inf_tol = translate ? force_tol : 1e300
    state = State(0, 0, false, copy(x0), normalised(copy(v0)))
    dE = point -> ask(state, point)[2]
    energy_of = point -> ask(state, point)[1]
    status = "stall"
    message = ""
    library = 0
    steps = 0
    energy = NaN
    gradient = Float64[]
    force_norm = NaN
    overlap = NaN
    curvature = NaN
    best_overlap = -1.0
    quiet = 0
    budget = iteration_budget(cap)
    previous_calls = -1
    previous_x = copy(state.x)
    previous_v = copy(state.v)

    try
        while !state.cap && state.calls < cap
            method = build_method(fields, inf_tol, tol_rot, budget, translate)
            local log
            try
                state.x, state.v, log = run!(method, energy_of, dE, state.x, state.v)
                state.v = normalised(state.v)
            catch err
                if err isa CapReached
                    state.cap = true
                    status = "cap"
                    break
                end
                status = "error"
                message = sprint(showerror, err)
                break
            end
            steps += 1
            library += library_count(log)
            if !isempty(ref)
                overlap = abs(dot(normalised(state.v), normalised(ref)))
            end
            if task == "mode" && isfinite(overlap) && overlap >= overlap_target
                status = "converged"
                break
            end
            if state.cap || state.calls >= cap
                status = "cap"
                break
            end
            probe_energy, probe_gradient = ask(state, state.x)
            energy = probe_energy
            gradient = probe_gradient
            force_norm = norm(probe_gradient)
            if translate && isfinite(force_norm) && force_norm <= force_tol
                status = "converged"
                break
            end
            residual = log_last(log, :res_trans)
            if residual === nothing
                residual = log_last(log, :maxres)
            end
            rotation = log_last(log, :res_rot)
            if translate && residual !== nothing && residual <= inf_tol && force_norm > force_tol
                inf_tol = max(inf_tol / 2, 1e-12)
            end
            if task == "mode" && rotation !== nothing && rotation <= tol_rot && overlap < overlap_target
                tol_rot = tol_rot / 10
            end
            moved = norm(state.x - previous_x) + norm(state.v - previous_v)
            if state.calls == previous_calls && moved < 1e-14
                quiet += 1
            else
                quiet = 0
            end
            previous_calls = state.calls
            previous_x = copy(state.x)
            previous_v = copy(state.v)
            if quiet >= 3
                status = "stall"
                break
            end
            if task == "mode" && isfinite(overlap)
                if overlap > best_overlap + 1e-5
                    best_overlap = overlap
                    quiet = 0
                elseif rotation !== nothing && rotation <= tol_rot && tol_rot < 1e-8
                    status = "stall"
                    break
                end
            end
            if steps > cap
                status = "stall"
                break
            end
        end
    catch err
        if err isa CapReached
            state.cap = true
            status = "cap"
        else
            status = "error"
            message = sprint(showerror, err)
        end
    end

    if state.cap
        status = "cap"
    end
    if status != "error" && !state.cap
        try
            if !isfinite(force_norm) && state.calls < cap
                energy, gradient = ask(state, state.x)
                force_norm = norm(gradient)
            end
            if !isempty(ref)
                overlap = abs(dot(normalised(state.v), normalised(ref)))
            end
        catch err
            if err isa CapReached
                state.cap = true
                status = "cap"
            else
                status = "error"
                message = sprint(showerror, err)
            end
        end
    end
    if state.cap || state.calls >= cap
        status = "cap"
    end
    if status == "converged" && translate && !(isfinite(force_norm) && force_norm <= force_tol)
        status = "stall"
    end
    write_done(status, message, state, energy, gradient, force_norm, overlap, curvature, library, steps)
    return nothing
end

function read_job()
    fields = Dict{String,String}()
    x = Float64[]
    v = Float64[]
    ref = Float64[]
    while !eof(stdin)
        line = readline()
        if line == ""
            continue
        end
        if line == "end"
            break
        end
        if line in ("x", "v", "ref")
            vector = read_vec()
            if line == "x"
                x = vector
            elseif line == "v"
                v = vector
            else
                ref = vector
            end
        else
            key, value = split(line, " ", limit=2)
            fields[key] = value
        end
    end
    return fields, x, v, ref
end

function main()
    println("ready")
    flush(stdout)
    while !eof(stdin)
        line = readline()
        if line == ""
            continue
        end
        if line == "bye"
            println("bye")
            flush(stdout)
            return
        end
        if line != "job"
            println(stderr, "ignored $line")
            continue
        end
        fields, x, v, ref = read_job()
        try
            run_job(fields, x, v, ref)
        catch err
            state = State(0, 0, false, x, v)
            write_done("error", sprint(showerror, err), state, NaN, Float64[], NaN, NaN, NaN, 0, 0)
        end
    end
end

main()
