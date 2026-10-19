# GeometryOptimization.jl driver for one OptBench minimization.
#
# Each stdin line is a job JSON object. The reply is one line,
# "OPTBENCH " followed by the start record. The potential is the
# Python harness callback in optbench.runners.geomoptjl_session.

import Logging
Logging.global_logger(Logging.ConsoleLogger(stderr, Logging.Error))

using AtomsBase
using AtomsCalculators
using GeometryOptimization
using LinearAlgebra
using Optim
using PythonCall
using StaticArrays
using Unitful
using UnitfulAtomic

const AC = AtomsCalculators

struct ForceCap <: Exception
    message::String
end

mutable struct HarnessCalc
    session::Py
    lunit
    eunit
    cap::Int
    last_key::Vector{Float64}
    last_positions::Vector{Float64}
    last_energy
    last_forces
    have::Bool
    evaluations::Int
    halt::Bool
    reason::String
end

function HarnessCalc(session::Py, lunit, eunit, cap::Int)
    HarnessCalc(session, lunit, eunit, cap, Float64[], Float64[], nothing, nothing, false, 0, false, "")
end

AC.energy_unit(calc::HarnessCalc) = calc.eunit
AC.length_unit(calc::HarnessCalc) = calc.lunit

function numeric_coords(system, lunit)
    n = length(system)
    coords = Vector{Float64}(undef, 3n)
    for i in 1:n
        pos = position(system, i)
        for axis in 1:3
            coords[3 * (i - 1) + axis] = Float64(ustrip(uconvert(lunit, pos[axis])))
        end
    end
    coords
end

function forces_from(grad::Vector{Float64}, funit)
    n = length(grad) ÷ 3
    FT = typeof(1.0 * funit)
    forces = Vector{SVector{3,FT}}(undef, n)
    for i in 1:n
        forces[i] = SVector{3,FT}(
            -grad[3i - 2] * funit,
            -grad[3i - 1] * funit,
            -grad[3i] * funit,
        )
    end
    forces
end

function evaluate!(calc::HarnessCalc, system)
    coords = numeric_coords(system, calc.lunit)
    if calc.have && coords == calc.last_key
        return calc.last_energy, calc.last_forces
    end
    result = calc.session.call(coords)
    if pyconvert(Bool, result["cap"])
        calc.reason = "cap"
        calc.halt = true
        throw(ForceCap("cap"))
    end
    err = pyconvert(String, result["error"])
    if !isempty(err)
        calc.reason = "crash"
        calc.halt = true
        throw(ForceCap(err))
    end
    energy_value = pyconvert(Float64, result["energy"])
    grad = pyconvert(Vector{Float64}, result["gradient"])
    if length(grad) != length(coords)
        calc.reason = "crash"
        calc.halt = true
        throw(ForceCap("gradient length does not match the coordinates"))
    end
    funit = calc.eunit / calc.lunit
    energy = energy_value * calc.eunit
    forces = forces_from(grad, funit)
    calc.last_key = coords
    calc.last_positions = copy(coords)
    calc.last_energy = energy
    calc.last_forces = forces
    calc.have = true
    calc.evaluations += 1
    return energy, forces
end

function AC.calculate(::AC.Energy, system, calc::HarnessCalc, parameters=nothing, state=nothing; kwargs...)
    energy, forces = evaluate!(calc, system)
    return (; energy, forces, state)
end

function AC.calculate(::AC.Forces, system, calc::HarnessCalc, parameters=nothing, state=nothing; kwargs...)
    energy, forces = evaluate!(calc, system)
    return (; energy, forces, state)
end

function force_l2(forces)::Float64
    isempty(forces) && return Inf
    acc = 0.0
    for force in forces
        for component in force
            value = Float64(ustrip(component))
            acc += value * value
        end
    end
    return sqrt(acc)
end

function make_callback(calc::HarnessCalc)
    function callback(_optim_state, state)
        calc.halt && return true
        calc.evaluations == 0 && return false
        calls = pyconvert(Int, calc.session.calls())
        if calls >= calc.cap
            calc.reason = "cap"
            calc.halt = true
            return true
        end
        met = pyconvert(Bool, calc.session.met(force_l2(state.forces)))
        if met
            calc.reason = "tolerance"
            calc.halt = true
            return true
        end
        return false
    end
    return callback
end

function build_system(kind::String, coords::Vector{Float64}, box::Vector{Float64})
    n = length(coords) ÷ 3
    if kind == "lj"
        lunit = u"bohr"
        eunit = u"hartree"
        atoms = [:H => SVector(coords[3i - 2], coords[3i - 1], coords[3i]) .* lunit for i in 1:n]
        cell = ([100.0, 0.0, 0.0] .* lunit, [0.0, 100.0, 0.0] .* lunit, [0.0, 0.0, 100.0] .* lunit)
        return periodic_system(atoms, cell), lunit, eunit
    elseif kind == "morse"
        if length(box) != 3
            error("morse job needs three box lengths")
        end
        lunit = u"Å"
        eunit = u"eV"
        atoms = [:H => SVector(coords[3i - 2], coords[3i - 1], coords[3i]) .* lunit for i in 1:n]
        cell = (
            [box[1], 0.0, 0.0] .* lunit,
            [0.0, box[2], 0.0] .* lunit,
            [0.0, 0.0, box[3]] .* lunit,
        )
        return periodic_system(atoms, cell), lunit, eunit
    else
        error("unknown potential kind $kind")
    end
end

function solver_for(method::String)
    if method == "lbfgs"
        return OptimLBFGS()
    elseif method == "cg"
        return OptimCG()
    elseif method == "gradientdescent"
        return OptimSD()
    else
        error("unknown method $method")
    end
end

function error_json(exc)::String
    msg = sprint(showerror, exc)
    msg = replace(msg, "\\" => "\\\\", "\"" => "\\\"")
    msg = replace(msg, "\n" => " ")
    if length(msg) > 500
        msg = first(msg, 500)
    end
    return "{\"schema\":\"optbench.start/v2\",\"success\":false,\"failure\":\"crash\",\"force_calls\":0,\"counter\":\"harness\",\"comparable\":true,\"reasons\":[\"crash\"],\"error\":\"$msg\"}"
end

function run_line(line::AbstractString)::String
    session_type = pyimport("optbench.runners.geomoptjl_session").Session
    session = session_type.from_json(line)
    kind = pyconvert(String, session.job_kind())
    coords = pyconvert(Vector{Float64}, session.job_coords())
    box = pyconvert(Vector{Float64}, session.job_box())
    method = pyconvert(String, session.job_method())
    cap = pyconvert(Int, session.job_cap())
    system, lunit, eunit = build_system(kind, coords, box)
    calc = HarnessCalc(session, lunit, eunit, cap)
    solver = solver_for(method)
    err = ""
    library_converged = false
    positions = coords
    started = time()
    try
        result = minimize_energy!(
            system,
            calc,
            solver;
            variablecell=false,
            maxiters=cap,
            maxtime=3600,
            tol_energy=Inf * eunit,
            tol_forces=1e-30 * (eunit / lunit),
            verbosity=0,
            callback=make_callback(calc),
        )
        positions = numeric_coords(result.system, lunit)
        library_converged = Bool(result.converged)
    catch exc
        if !isempty(calc.last_positions)
            positions = calc.last_positions
        end
        if exc isa ForceCap
            if exc.message == "cap"
                calc.reason = "cap"
            else
                calc.reason = isempty(calc.reason) ? "crash" : calc.reason
                err = exc.message
            end
        else
            calc.reason = isempty(calc.reason) ? "crash" : calc.reason
            err = sprint(showerror, exc)
        end
    end
    elapsed = time() - started
    reason = isempty(calc.reason) ? "" : calc.reason
    return pyconvert(String, session.finish(positions, reason, err, elapsed, library_converged))
end

function main()
    version = pkgversion(GeometryOptimization)
    optim_version = pkgversion(Optim)
    println(stdout, "OPTBENCH ready $version $optim_version")
    flush(stdout)
    for line in eachline(stdin)
        text = strip(line)
        isempty(text) && continue
        text == "OPTBENCH quit" && break
        reply = try
            run_line(text)
        catch exc
            error_json(exc)
        end
        println(stdout, "OPTBENCH ", reply)
        flush(stdout)
    end
end

main()
