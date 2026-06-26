// BicepCurlFormScreen.tsx
//
// Lets the user upload an expert's bicep-curl video and their own, runs both
// through MoveNet, aligns the two with DTW, and shows an overlaid elbow-angle
// chart plus a form score with a per-joint breakdown.
//
// ASSUMPTION: both videos are filmed from a similar camera angle and
// distance — see formScoreUtils.ts for why that matters and what would need
// to change if that assumption is relaxed later.

import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
  Dimensions,
  ScrollView,
} from 'react-native';
import { launchImageLibrary } from 'react-native-image-picker';
import { FFmpegKit, ReturnCode } from '@wokcito/ffmpeg-kit-react-native';
import RNFS from 'react-native-fs';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { LineChart } from 'react-native-chart-kit';
import * as base64js from 'base64-js';

import {
  BICEP_CURL_CONFIG,
  FramePose,
  FormScoreResult,
  computeFormScore,
} from '../utils/formScoreUtils';

const FPS = 8;
const FRAME_SIZE = 192;
const RAW_FRAME_BYTES = FRAME_SIZE * FRAME_SIZE * 3; // rgb24

const USER_COLOR = '#3878D8';
const EXPERT_COLOR = '#D8782D';

// Extracts frames from a video and runs the pose model on each one.
// Returns an array of {tSec, keypoints} — no scoring logic lives here.
async function extractFramePoses(
  uri: string,
  model: any, // TFLite model instance from useTensorflowModel
  onProgress: (msg: string) => void,
): Promise<FramePose[]> {
  const tempDir = `${RNFS.CachesDirectoryPath}/frames_${Date.now()}_${Math.floor(
    Math.random() * 1e6,
  )}`;
  await RNFS.mkdir(tempDir);

  try {
    onProgress('Extracting frames...');
    const ffmpegCommand = `-i "${uri}" -vf "fps=${FPS},scale=${FRAME_SIZE}:${FRAME_SIZE}" -f image2 -pix_fmt rgb24 "${tempDir}/frame_%04d.raw"`;
    const session = await FFmpegKit.execute(ffmpegCommand);
    const returnCode = await session.getReturnCode();
    if (!ReturnCode.isSuccess(returnCode)) {
      throw new Error('Failed to extract frames from the video.');
    }

    const files = await RNFS.readDir(tempDir);
    const rawFiles = files
      .filter(f => f.name.endsWith('.raw'))
      .sort((a, b) => a.name.localeCompare(b.name));

    const frames: FramePose[] = [];
    for (let i = 0; i < rawFiles.length; i++) {
      onProgress(`Analyzing frame ${i + 1}/${rawFiles.length}...`);
      const base64Str = await RNFS.readFile(rawFiles[i].path, 'base64');
      const buffer = base64js.toByteArray(base64Str);
      if (buffer.length !== RAW_FRAME_BYTES) continue; // skip malformed frame

      const outputs = await model.run([buffer.buffer]);
      // IMPORTANT: this is a *copy*, not a view. `new Float32Array(arrayBuffer)`
      // only wraps the runtime's output buffer — many TFLite-style runtimes
      // reuse that same buffer on every model.run() call for performance. If we
      // stored that view directly, every frame in `frames` would end up pointing
      // at whatever the LAST inference call wrote, since we read angles from
      // this array later rather than immediately. Float32Array.from() copies
      // the values into an independent array right now, so each frame keeps its
      // own snapshot.
      const outputView = new Float32Array(outputs[0] as ArrayBuffer);
      const keypoints = Float32Array.from(outputView);
      frames.push({ tSec: i / FPS, keypoints });
    }
    return frames;
  } finally {
    await RNFS.unlink(tempDir).catch(() => {});
  }
}

function jointLabel(name: string): string {
  return BICEP_CURL_CONFIG.joints.find(j => j.name === name)?.label ?? name;
}

function scoreColor(score: number): string {
  if (score >= 85) return '#1D9E75';
  if (score >= 65) return '#BA7517';
  return '#E24B4A';
}

export default function BicepCurlFormScreen() {
  const [expertUri, setExpertUri] = useState<string | null>(null);
  const [userUri, setUserUri] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FormScoreResult | null>(null);

  const plugin = useTensorflowModel(require('../assets/movenet_lightning.tflite'), []);
  const actualModel = plugin.state === 'loaded' ? plugin.model : undefined;

  const pickVideo = async (role: 'expert' | 'user') => {
    const pickResult = await launchImageLibrary({ mediaType: 'video' });
    const uri = pickResult.assets?.[0]?.uri;
    if (!uri) return;
    if (role === 'expert') setExpertUri(uri);
    else setUserUri(uri);
    setResult(null);
    setError(null);
  };

  const handleAnalyze = async () => {
    if (!actualModel || !expertUri || !userUri) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const expertFrames = await extractFramePoses(expertUri, actualModel, msg =>
        setStatus(`Expert video — ${msg}`),
      );
      if (expertFrames.length < 5) {
        throw new Error('Could not detect a clear pose in the expert video.');
      }

      const userFrames = await extractFramePoses(userUri, actualModel, msg =>
        setStatus(`Your video — ${msg}`),
      );
      if (userFrames.length < 5) {
        throw new Error('Could not detect a clear pose in your video.');
      }

      setStatus('Aligning and scoring...');
      const formResult = computeFormScore(userFrames, expertFrames, BICEP_CURL_CONFIG);
      setResult(formResult);
    } catch (e: any) {
      console.error(e);
      setError(e?.message ?? 'Something went wrong while analyzing the videos.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setExpertUri(null);
    setUserUri(null);
    setResult(null);
    setError(null);
    setStatus('');
  };

  const canAnalyze = !!actualModel && !!expertUri && !!userUri && !loading;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.title}>Bicep Curl Form Score</Text>

      <View style={styles.uploadRow}>
        <UploadCard
          label="Expert video"
          selected={!!expertUri}
          disabled={loading}
          onPress={() => pickVideo('expert')}
        />
        <UploadCard
          label="Your video"
          selected={!!userUri}
          disabled={loading}
          onPress={() => pickVideo('user')}
        />
      </View>

      <Text style={styles.tipText}>
        Tip: film both videos from the same side and distance, and trim each to a single
        rep for the most accurate score.
      </Text>

      <TouchableOpacity
        style={[styles.analyzeButton, !canAnalyze && styles.analyzeButtonDisabled]}
        onPress={handleAnalyze}
        disabled={!canAnalyze}
      >
        <Text style={styles.analyzeButtonText}>
          {!actualModel ? 'Loading model...' : 'Analyze form'}
        </Text>
      </TouchableOpacity>

      {loading && (
        <View style={styles.statusContainer}>
          <ActivityIndicator size="large" color={USER_COLOR} />
          <Text style={styles.statusText}>{status}</Text>
        </View>
      )}

      {!!error && (
        <View style={styles.errorBox}>
          <Text style={styles.errorText}>{error}</Text>
        </View>
      )}

      {!loading && result && (
        <View style={styles.resultContainer}>
          <View style={styles.legendRow}>
            <LegendItem color={USER_COLOR} label="You" />
            <LegendItem color={EXPERT_COLOR} label="Expert" />
          </View>

          <ScrollView horizontal showsHorizontalScrollIndicator style={styles.scrollView}>
            <LineChart
              data={{
                labels: result.chart.labels,
                datasets: [
                  { data: result.chart.userValues, color: () => USER_COLOR, strokeWidth: 2 },
                  { data: result.chart.expertValues, color: () => EXPERT_COLOR, strokeWidth: 2 },
                ],
              }}
              width={Math.max(
                Dimensions.get('window').width - 32,
                result.chart.userValues.length * 14,
              )}
              height={220}
              yAxisSuffix="°"
              chartConfig={{
                backgroundColor: '#ffffff',
                backgroundGradientFrom: '#ffffff',
                backgroundGradientTo: '#ffffff',
                decimalPlaces: 0,
                color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                style: { borderRadius: 16 },
                propsForDots: { r: '0' },
              }}
              style={{ marginVertical: 8, borderRadius: 16 }}
              withShadow={false}
              bezier
            />
          </ScrollView>

          <View style={styles.scoreCard}>
            <Text style={styles.scoreLabel}>Form score</Text>
            <Text style={[styles.scoreValue, { color: scoreColor(result.overallScore) }]}>
              {result.overallScore}
              <Text style={styles.scoreOutOf}>/100</Text>
            </Text>
          </View>

          <View style={styles.breakdownCard}>
            {Object.entries(result.jointScores).map(([name, score]) => (
              <View key={name} style={styles.jointRow}>
                <Text style={styles.jointLabel}>{jointLabel(name)}</Text>
                <Text style={[styles.jointScore, { color: scoreColor(score) }]}>{score}</Text>
              </View>
            ))}
          </View>

          {result.userSide !== result.expertSide && (
            <Text style={styles.warningText}>
              Note: your {result.userSide} arm was tracked against the expert's{' '}
              {result.expertSide} arm. For best accuracy, film both videos facing the same
              direction.
            </Text>
          )}

          <TouchableOpacity style={styles.resetButton} onPress={handleReset}>
            <Text style={styles.resetButtonText}>Analyze another rep</Text>
          </TouchableOpacity>
        </View>
      )}
    </ScrollView>
  );
}

function UploadCard({
  label,
  selected,
  disabled,
  onPress,
}: {
  label: string;
  selected: boolean;
  disabled: boolean;
  onPress: () => void;
}) {
  return (
    <TouchableOpacity
      style={[styles.uploadCard, selected && styles.uploadCardSelected]}
      onPress={onPress}
      disabled={disabled}
    >
      <Text style={styles.uploadCardLabel}>{label}</Text>
      <Text style={styles.uploadCardStatus}>{selected ? 'Selected ✓' : 'Tap to select'}</Text>
    </TouchableOpacity>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <View style={styles.legendItem}>
      <View style={[styles.legendDot, { backgroundColor: color }]} />
      <Text style={styles.legendLabel}>{label}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#fff' },
  content: { padding: 16, paddingBottom: 40, alignItems: 'center' },
  title: { fontSize: 24, fontWeight: 'bold', marginBottom: 20, marginTop: 20 },

  uploadRow: { flexDirection: 'row', width: '100%', gap: 12 },
  uploadCard: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#ddd',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  uploadCardSelected: { borderColor: USER_COLOR, backgroundColor: '#EEF3FB' },
  uploadCardLabel: { fontSize: 15, fontWeight: '600', marginBottom: 6, textAlign: 'center' },
  uploadCardStatus: { fontSize: 13, color: '#666' },

  tipText: { fontSize: 12, color: '#888', textAlign: 'center', marginTop: 12, marginBottom: 8 },

  analyzeButton: {
    backgroundColor: USER_COLOR,
    borderRadius: 10,
    paddingVertical: 14,
    paddingHorizontal: 24,
    width: '100%',
    alignItems: 'center',
    marginTop: 8,
  },
  analyzeButtonDisabled: { backgroundColor: '#aac0e0' },
  analyzeButtonText: { color: '#fff', fontSize: 16, fontWeight: '600' },

  statusContainer: { marginTop: 30, alignItems: 'center' },
  statusText: { marginTop: 10, fontSize: 15, color: '#333', textAlign: 'center' },

  errorBox: {
    marginTop: 20,
    padding: 14,
    borderRadius: 10,
    backgroundColor: '#FCEBEB',
    width: '100%',
  },
  errorText: { color: '#A32D2D', fontSize: 14, textAlign: 'center' },

  resultContainer: { width: '100%', marginTop: 30, alignItems: 'center' },

  legendRow: { flexDirection: 'row', gap: 20, marginBottom: 4 },
  legendItem: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  legendDot: { width: 10, height: 10, borderRadius: 5 },
  legendLabel: { fontSize: 13, color: '#333' },

  scrollView: { width: '100%' },

  scoreCard: { alignItems: 'center', marginTop: 20 },
  scoreLabel: { fontSize: 14, color: '#666', marginBottom: 4 },
  scoreValue: { fontSize: 48, fontWeight: 'bold' },
  scoreOutOf: { fontSize: 20, color: '#999', fontWeight: 'normal' },

  breakdownCard: { width: '100%', marginTop: 24, gap: 10 },
  jointRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 10,
    backgroundColor: '#f5f5f5',
  },
  jointLabel: { fontSize: 14, color: '#333', flex: 1 },
  jointScore: { fontSize: 16, fontWeight: '700' },

  warningText: {
    fontSize: 12,
    color: '#854F0B',
    textAlign: 'center',
    marginTop: 16,
    paddingHorizontal: 8,
  },

  resetButton: { marginTop: 24, paddingVertical: 10, paddingHorizontal: 20 },
  resetButtonText: { color: USER_COLOR, fontSize: 14, fontWeight: '600' },
});