"""Runtime helpers independent of CEE's edge/pseudo-label decisions."""
import torch


def autocast_settings(cfg):
    enabled = bool(cfg.get('amp', True)) and torch.cuda.is_available()
    name = cfg.get('amp_dtype', 'auto')
    if name not in ('auto', 'bf16', 'fp16'):
        raise ValueError('amp_dtype must be auto, bf16 or fp16')
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if enabled and name == 'bf16' and not bf16:
        raise ValueError('bf16 requested but unsupported; use auto or fp16')
    dtype = torch.bfloat16 if name == 'bf16' or (name == 'auto' and bf16) else torch.float16
    return enabled, dtype


def loader_options(cfg, validation=False):
    workers = int(cfg.get('val_num_workers' if validation else 'num_workers', 1 if validation else 4))
    kwargs = dict(num_workers=workers, pin_memory=True)
    if workers > 0:
        kwargs.update(persistent_workers=bool(cfg.get('persistent_workers', True)),
                      prefetch_factor=int(cfg.get('prefetch_factor', 2)))
    return kwargs


@torch.no_grad()
def update_ema(student, teacher, decay):
    student = student.module if hasattr(student, 'module') else student
    params = dict(student.named_parameters())
    for name, dest in teacher.named_parameters():
        dest.lerp_(params[name].detach(), 1 - decay)
    buffers = dict(student.named_buffers())
    for name, dest in teacher.named_buffers():
        source = buffers[name].detach()
        if dest.is_floating_point():
            dest.lerp_(source, 1 - decay)
        else:
            dest.copy_(source)


def load_segmentation_state(model, state, allow_missing_ctsa=False):
    """Strict for backbone/head; only the known training-only namespace is optional."""
    model = model.module if hasattr(model, 'module') else model
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    if getattr(model, 'ctsa', None) is None:
        state = {k: v for k, v in state.items() if not k.startswith('ctsa.')}
    missing, extra = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not (allow_missing_ctsa and k.startswith('ctsa.'))]
    if bad_missing or extra:
        raise RuntimeError(f'Checkpoint mismatch: missing={bad_missing}, unexpected={extra}')


class StepTimer:
    """Diagnostic mode only; normal iterations do not synchronize the GPU."""
    def __init__(self, enabled):
        self.enabled = enabled
        self.events = []
        if enabled:
            self.mark('start')

    def mark(self, name):
        if self.enabled:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.events.append((name, event))

    def finish(self):
        if not self.enabled:
            return {}
        torch.cuda.synchronize()
        return {name: prev.elapsed_time(event)
                for (_, prev), (name, event) in zip(self.events, self.events[1:])}


def intersection_union(pred, target, classes):
    pred, target = pred.reshape(-1), target.reshape(-1)
    valid = target != 255
    pred, target = pred[valid], target[valid]
    intersection = torch.bincount(pred[pred == target], minlength=classes)[:classes]
    output = torch.bincount(pred, minlength=classes)[:classes]
    truth = torch.bincount(target, minlength=classes)[:classes]
    return intersection, output + truth - intersection
