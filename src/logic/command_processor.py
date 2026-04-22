"""
command_processor.py
Handles all user input (keyboard keys) for the SYNAPSE system.

Design principles:
  - Stateless per-call: receives a CommandContext, returns a CommandResult.
  - No string building: delegates all NL generation to OutputGenerator.
  - No direct TTS calls: speaks via ctx.tts so callers can swap the engine.
  - Fully testable without a camera or display.

Usage (inside main.py's run loop):
    result = self.cmd_processor.handle(key, self._build_context(frame, tracks))
    if result.should_quit:
        break
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CommandContext:
    """All live state the processor needs to execute a command.

    Attributes:
        key          : OpenCV waitKey byte (e.g. ord('w'))
        frame        : Current BGR frame (numpy array). May be None.
        tracks       : Latest track list from ByteTracker.
        tts          : TTSOutput instance.
        output_gen   : OutputGenerator instance.
        async_proc   : AsyncProcessor instance.
        scene_memory : SceneMemory instance.
        captioner    : SceneCaptioner instance.
        ocr          : OCRModule instance.
        distance_est : DistanceEstimator instance.
        frame_width  : Camera frame width (from config).
        warnings_enabled : Current warnings toggle state (mutable via result).
    """
    key: int
    frame: Any                          # np.ndarray — avoid numpy import here
    tracks: List[Dict]
    tts: Any                            # TTSOutput
    output_gen: Any                     # OutputGenerator
    async_proc: Any                     # AsyncProcessor
    scene_memory: Any                   # SceneMemory
    captioner: Any                      # SceneCaptioner
    ocr: Any                            # OCRModule
    distance_est: Any                   # DistanceEstimator
    frame_width: int
    warnings_enabled: bool


@dataclass
class CommandResult:
    """Returned by CommandProcessor.handle() after every key press.

    Attributes:
        should_quit         : True only when the user pressed Q.
        warnings_enabled    : Updated toggle state (may be unchanged).
        handled             : True if the key mapped to a known command.
        message             : Human-readable log string for the action taken.
    """
    should_quit: bool = False
    warnings_enabled: bool = True
    handled: bool = False
    message: str = ""


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

class CommandProcessor:
    """Dispatches keyboard commands to the correct SYNAPSE subsystem.

    All command logic that was previously inline in main.py's run() loop
    lives here. Each command is a private method named _cmd_<key>().
    """

    # Keys we recognise
    KEY_WHATS_AHEAD = ord('w')
    KEY_DESCRIBE    = ord('d')
    KEY_READ_TEXT   = ord('r')
    KEY_TOGGLE_WARN = ord('t')
    KEY_QUIT        = ord('q')

    def __init__(self):
        print("COMMAND PROCESSOR: Initialized")
        self._dispatch = {
            self.KEY_WHATS_AHEAD: self._cmd_whats_ahead,
            self.KEY_DESCRIBE:    self._cmd_describe,
            self.KEY_READ_TEXT:   self._cmd_read_text,
            self.KEY_TOGGLE_WARN: self._cmd_toggle_warnings,
            self.KEY_QUIT:        self._cmd_quit,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle(self, ctx: CommandContext) -> CommandResult:
        """Process one keypress.

        Args:
            ctx: Full system context at the time of the keypress.
        Returns:
            CommandResult with updated state and quit flag.
        """
        handler = self._dispatch.get(ctx.key)
        if handler is None:
            # Unknown key — no-op, pass warnings state through unchanged
            return CommandResult(warnings_enabled=ctx.warnings_enabled)
        return handler(ctx)

    # ------------------------------------------------------------------
    # W — What's ahead?
    # ------------------------------------------------------------------

    def _cmd_whats_ahead(self, ctx: CommandContext) -> CommandResult:
        """Report current scene based on scene memory diff."""
        logger.info("USER: What's ahead?")

        if not ctx.tracks:
            # No live detections — check if anything recently disappeared
            changes = ctx.scene_memory.get_changes(
                [], ctx.frame_width, ctx.distance_est
            )
            summary = ctx.output_gen.format_scene_changes_empty(changes)
            ctx.scene_memory.update([], ctx.frame_width, ctx.distance_est)
        else:
            changes = ctx.scene_memory.get_changes(
                ctx.tracks, ctx.frame_width, ctx.distance_est
            )
            summary = ctx.output_gen.format_scene_changes(
                changes, ctx.distance_est
            )
            ctx.scene_memory.update(
                ctx.tracks, ctx.frame_width, ctx.distance_est
            )

        logger.info(f"SYSTEM: {summary}")
        ctx.tts.speak(summary, blocking=False)

        return CommandResult(
            warnings_enabled=ctx.warnings_enabled,
            handled=True,
            message=summary,
        )

    # ------------------------------------------------------------------
    # D — Describe scene (BLIP)
    # ------------------------------------------------------------------

    def _cmd_describe(self, ctx: CommandContext) -> CommandResult:
        """Run BLIP captioning + spatial layout asynchronously."""
        if ctx.async_proc.is_running:
            ctx.tts.speak("Still analyzing, please wait.", blocking=False)
            return CommandResult(
                warnings_enabled=ctx.warnings_enabled,
                handled=True,
                message="Describe requested but async processor busy.",
            )

        logger.info("USER: Describe scene")
        ctx.tts.speak("Analyzing scene.", blocking=False)

        # Capture snapshots so the async task uses a consistent frame/tracks
        current_frame  = ctx.frame.copy()
        current_tracks = list(ctx.tracks)  # shallow copy is enough

        # Import here to keep this module free of heavy vision imports
        from src.vision.spatial_layout import build_spatial_summary

        def describe_task():
            caption   = ctx.captioner.generate_caption(current_frame, verbose=False)
            formatted = ctx.output_gen.format_caption(caption)
            spatial   = build_spatial_summary(
                current_tracks, ctx.frame_width, ctx.distance_est
            )
            return f"{formatted} {spatial}" if current_tracks else formatted

        def on_done(result: str):
            logger.info(f"SYSTEM: {result}")
            ctx.tts.speak(result, blocking=False)

        ctx.async_proc.run(describe_task, callback=on_done)

        return CommandResult(
            warnings_enabled=ctx.warnings_enabled,
            handled=True,
            message="Describe task dispatched.",
        )

    # ------------------------------------------------------------------
    # R — Read text (OCR)
    # ------------------------------------------------------------------

    def _cmd_read_text(self, ctx: CommandContext) -> CommandResult:
        """Run EasyOCR asynchronously on the current frame."""
        if ctx.async_proc.is_running:
            ctx.tts.speak("Still reading, please wait.", blocking=False)
            return CommandResult(
                warnings_enabled=ctx.warnings_enabled,
                handled=True,
                message="OCR requested but async processor busy.",
            )

        logger.info("USER: Read text")
        ctx.tts.speak("Reading text.", blocking=False)

        current_frame = ctx.frame.copy()

        def ocr_task():
            text, _ = ctx.ocr.extract_text(current_frame, verbose=False)
            return ctx.output_gen.format_ocr_result(text)

        def on_done(result: str):
            logger.info(f"SYSTEM: {result}")
            ctx.tts.speak(result, blocking=False)

        ctx.async_proc.run(ocr_task, callback=on_done)

        return CommandResult(
            warnings_enabled=ctx.warnings_enabled,
            handled=True,
            message="OCR task dispatched.",
        )

    # ------------------------------------------------------------------
    # T — Toggle warnings
    # ------------------------------------------------------------------

    def _cmd_toggle_warnings(self, ctx: CommandContext) -> CommandResult:
        """Flip the proactive collision warning flag."""
        new_state = not ctx.warnings_enabled
        status    = "enabled" if new_state else "disabled"

        logger.info(f"SYSTEM: Warnings {status}")
        ctx.tts.speak(f"Warnings {status}.", blocking=False)

        return CommandResult(
            warnings_enabled=new_state,
            handled=True,
            message=f"Warnings toggled to {status}.",
        )

    # ------------------------------------------------------------------
    # Q — Quit
    # ------------------------------------------------------------------

    def _cmd_quit(self, ctx: CommandContext) -> CommandResult:
        """Signal the run loop to exit cleanly."""
        logger.info("SYSTEM: Shutting down.")
        return CommandResult(
            should_quit=True,
            warnings_enabled=ctx.warnings_enabled,
            handled=True,
            message="Quit command received.",
        )