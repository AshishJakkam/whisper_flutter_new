import "dart:convert";
import "dart:ffi";
import "dart:io";
import "dart:isolate";

import "package:ffi/ffi.dart";
import "package:whisper_flutter_new/bean/_models.dart";
import "package:whisper_flutter_new/download_model.dart";

typedef _SessionRequestNative = Pointer<Char> Function(Pointer<Char>);
typedef _SessionRequestDart = Pointer<Char> Function(Pointer<Char>);

const String _symbol = "session_request";

DynamicLibrary _openLib() =>
    Platform.isAndroid ? DynamicLibrary.open("libwhisper.so") : DynamicLibrary.process();

/// Calls the native entry point on a background isolate.
///
/// The lookup happens inside the isolate: an FFI function object belongs to the isolate that
/// resolved it and cannot be sent to another one.
Future<Object?> _callNative(String body) {
  return Isolate.run(() {
    final send = _openLib().lookupFunction<_SessionRequestNative, _SessionRequestDart>(_symbol);
    final native = body.toNativeUtf8();
    final Pointer<Char> res = send(native.cast<Char>());
    try {
      return json.decode(res.cast<Utf8>().toDartString());
    } finally {
      malloc.free(native);
      malloc.free(res);
    }
  });
}

bool _isAvailable() {
  try {
    _openLib().lookup<NativeFunction<_SessionRequestNative>>(_symbol);
    return true;
  } catch (_) {
    return false;
  }
}

/// A whisper model loaded once and reused for every transcription.
///
/// `Whisper` initialises the model, runs one inference and frees it, so every call pays the whole
/// read of the model file. Measured on an iPhone 16 Pro that was ~12 s per call and the only reason
/// a 5 s live window took over 13 s. With the context kept, a window costs its own audio length
/// instead of the model size.
class WhisperSession {
  WhisperSession({required this.model, required this.modelPath, this.threads = 6});

  final WhisperModel model;
  final String modelPath;
  final int threads;

  /// False on a platform whose binary predates the session API; callers then use `Whisper`.
  static bool get isSupported => _isAvailable();

  bool _open = false;

  bool get isOpen => _open;

  /// Loads the model. Idempotent and cheap once open.
  Future<bool> open() async {
    if (_open) return true;
    try {
      final result = await _callNative(json.encode({"@type": "sessionOpen", "model": modelPath}));
      _open = result is Map && result["ok"] == true;
    } catch (_) {
      _open = false;
    }
    return _open;
  }

  /// Runs one inference on the retained model, or null when it could not be loaded.
  Future<WhisperTranscribeResponse?> transcribe({required TranscribeRequest request}) async {
    if (!await open()) return null;
    final payload = TranscribeRequestDto.fromTranscribeRequest(request, modelPath).toRequestString();
    final result = await _callNative(payload);
    if (result is! Map || result["text"] == null) {
      throw Exception(result is Map ? result["message"] : "session returned no result");
    }
    return WhisperTranscribeResponse.fromJson(Map<String, dynamic>.from(result));
  }

  /// Frees the model. Safe to call more than once.
  Future<void> close() async {
    if (!_open) return;
    _open = false;
    try {
      await _callNative(json.encode({"@type": "sessionClose"}));
    } catch (_) {}
  }
}
