import { useState, useCallback } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import {
  Camera,
  useCameraDevices,
  useCameraPermission,
  useFrameOutput,
} from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { runOnJS } from 'react-native-worklets';
import { useResizer } from 'react-native-vision-camera-resizer';

const RIGHT_SHOULDER = 6;
const RIGHT_ELBOW = 8;
const RIGHT_WRIST = 10;
const MIN_CONFIDENCE = 0.3;

function calculateAngle(
  a: { x: number; y: number },
  b: { x: number; y: number },
  c: { x: number; y: number },
): number {
  'worklet';
  const radians =
    Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180) / Math.PI);
  if (angle > 180) angle = 360 - angle;
  return angle;
}

type Point2D = { x: number; y: number };

const Line = ({ p1, p2, color }: { p1: Point2D; p2: Point2D, color: string }) => {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const length = Math.sqrt(dx * dx + dy * dy);
  const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
  return (
    <View
      style={{
        position: 'absolute',
        left: (p1.x + p2.x) / 2 - length / 2,
        top: (p1.y + p2.y) / 2 - 2,
        width: length,
        height: 4,
        backgroundColor: color,
        transform: [{ rotate: `${angle}deg` }],
      }}
    />
  );
};

const Point = ({ p }: { p: Point2D }) => (
  <View
    style={{
      position: 'absolute',
      left: p.x - 6,
      top: p.y - 6,
      width: 12,
      height: 12,
      borderRadius: 6,
      backgroundColor: 'red',
      borderWidth: 2,
      borderColor: 'white',
    }}
  />
);

export default function App() {
  const devices = useCameraDevices();
  const device = devices.find(d => d.position === 'front');
  const { hasPermission, requestPermission } = useCameraPermission();
  
  const [layout, setLayout] = useState({ width: 1, height: 1 });
  const [pose, setPose] = useState<{
    shoulder: Point2D;
    elbow: Point2D;
    wrist: Point2D;
    angle: number;
    score: number;
    frameWidth: number;
    frameHeight: number;
  } | null>(null);

  const plugin = useTensorflowModel(
    require('./assets/movenet_lightning.tflite'),
    [],
  );

  const actualModel = plugin.state === 'loaded' ? plugin.model : undefined;

  const resizerState = useResizer({
    width: 192,
    height: 192,
    channelOrder: 'rgb',
    dataType: 'uint8',
    scaleMode: 'cover',
    pixelLayout: 'interleaved',
  });

  const actualResizer = resizerState.state === 'ready' ? resizerState.resizer : undefined;

  const updatePose = useCallback((newPose: any) => {
    setPose(newPose);
  }, []);

  const frameOutput = useFrameOutput({
    pixelFormat: 'yuv',
    enablePhysicalBufferRotation: true,
    onFrame(frame) {
      'worklet';
      if (actualResizer == null || actualModel == null) {
        frame.dispose();
        return;
      }

      try {
        const resized = actualResizer.resize(frame);
        const input = resized.getPixelBuffer();
        resized.dispose();

        const outputs = actualModel.runSync([input]);
        const kp = new Float32Array(outputs[0]);

        const shoulder = { y: kp[RIGHT_SHOULDER * 3], x: kp[RIGHT_SHOULDER * 3 + 1], score: kp[RIGHT_SHOULDER * 3 + 2] };
        const elbow = { y: kp[RIGHT_ELBOW * 3], x: kp[RIGHT_ELBOW * 3 + 1], score: kp[RIGHT_ELBOW * 3 + 2] };
        const wrist = { y: kp[RIGHT_WRIST * 3], x: kp[RIGHT_WRIST * 3 + 1], score: kp[RIGHT_WRIST * 3 + 2] };

        const angle = calculateAngle(shoulder, elbow, wrist);
        const minScore = Math.min(shoulder.score, elbow.score, wrist.score);

        runOnJS(updatePose)({
          shoulder: { x: shoulder.x, y: shoulder.y },
          elbow: { x: elbow.x, y: elbow.y },
          wrist: { x: wrist.x, y: wrist.y },
          angle,
          score: minScore,
          frameWidth: frame.width,
          frameHeight: frame.height,
        });
      } catch (e) {
        console.log('frame processing error', e);
      }

      frame.dispose();
    },
  });

  const renderSkeleton = () => {
    if (!pose || !device) return null;

    const { frameWidth, frameHeight, score } = pose;
    const cropSize = Math.min(frameWidth, frameHeight);
    const scale = Math.max(layout.width / frameWidth, layout.height / frameHeight);
    const screenCropSize = cropSize * scale;
    const isFront = device.position === 'front';

    const mapPoint = (p: Point2D) => {
      const mappedX = layout.width / 2 + screenCropSize * (p.x - 0.5);
      const mappedY = layout.height / 2 + screenCropSize * (p.y - 0.5);
      return {
        x: isFront ? layout.width - mappedX : mappedX,
        y: mappedY
      };
    };

    const pS = mapPoint(pose.shoulder);
    const pE = mapPoint(pose.elbow);
    const pW = mapPoint(pose.wrist);
    
    // Use red for low confidence, lime for good confidence
    const lineColor = score > MIN_CONFIDENCE ? 'lime' : 'red';

    return (
      <View style={StyleSheet.absoluteFill} pointerEvents="none">
        <Line p1={pS} p2={pE} color={lineColor} />
        <Line p1={pE} p2={pW} color={lineColor} />
        <Point p={pS} />
        <Point p={pE} />
        <Point p={pW} />
      </View>
    );
  };

  if (!hasPermission) {
    return (
      <View style={styles.center}>
        <Text onPress={requestPermission}>Grant Camera Permission</Text>
      </View>
    );
  }

  if (!device) return <Text>No Camera Found</Text>;

  if (plugin.state !== 'loaded') {
    return (
      <View style={styles.center}>
        <Text>Model state: {plugin.state}</Text>
      </View>
    );
  }

  if (resizerState.state !== 'ready') {
    return (
      <View style={styles.center}>
        <Text>Resizer state: {resizerState.state}</Text>
      </View>
    );
  }

  return (
    <View 
      style={{ flex: 1 }} 
      onLayout={(e) => setLayout(e.nativeEvent.layout)}
    >
      <Camera
        style={{ flex: 1 }}
        device={device}
        isActive={true}
        outputs={[frameOutput]}
      />
      {renderSkeleton()}
      <View style={styles.overlay}>
        <Text style={styles.angleText}>
          Elbow: {pose ? pose.angle.toFixed(0) : '--'}°
        </Text>
        <Text style={{ color: 'white', fontSize: 16 }}>
          Conf: {pose ? pose.score.toFixed(2) : '--'}
        </Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  overlay: {
    position: 'absolute',
    top: 50,
    left: 20,
    backgroundColor: 'rgba(0,0,0,0.5)',
    padding: 10,
    borderRadius: 8,
  },
  angleText: { color: 'red', fontSize: 24, fontWeight: 'bold' },
});