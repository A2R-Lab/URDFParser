"""Typed exceptions for the URDF parser.

Historically a malformed / unsupported URDF caused the parser to
``print(...)`` and call ``exit()`` -- a bare ``SystemExit`` that kills the
whole host process with no traceback and is uncatchable by normal
``except Exception`` handlers. These typed exceptions replace that failure
mode so callers (the equivalence harness, GRiD codegen, downstream tools)
can catch and surface a structured, debuggable error instead.

All parser-raised errors derive from :class:`URDFParseError` so a caller may
catch the whole family with a single ``except URDFParseError``.
"""


class URDFParseError(Exception):
    """Base class for every error raised while parsing a URDF model."""


class MimicResolutionError(URDFParseError, ValueError):
    """Raised when a ``<mimic>`` relation cannot be resolved (unknown target
    joint or a mimic cycle). Derives from ValueError too so pre-existing
    ``except ValueError`` callers keep working, while the URDFParseError base
    lets it propagate through ``parse()`` instead of degrading to None."""


class UnsupportedJointTypeError(URDFParseError):
    """Raised when a URDF joint declares a type the parser cannot model.

    Carries the offending joint type (and optionally the joint name) so a
    caller can report exactly which joint tripped the parser.
    """

    def __init__(self, jtype, joint_name=None):
        self.jtype = jtype
        self.joint_name = joint_name
        # Everything the parser models: revolute/continuous/prismatic/fixed/
        # floating are Tier-A cardinal; helical (screw) parses natively;
        # planar and translation DECOMPOSE at parse time into cardinal
        # sub-joints; spherical parses as a native 3-DoF (NQ != NV) joint.
        # RBDReference models all of these; downstream CUDA codegen gates any
        # not-yet-ported algorithm with its own clear error.
        supported = ("revolute, continuous, prismatic, fixed, floating, "
                     "helical (screw), planar, translation, spherical")
        where = f" (joint '{joint_name}')" if joint_name is not None else ""
        super().__init__(
            f"Unsupported joint type '{jtype}'{where}. "
            f"Supported joint types are: {supported}."
        )
