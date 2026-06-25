import React, { useState } from 'react';
import { View, Text, Button, StyleSheet, ActivityIndicator, Dimensions, ScrollView } from 'react-native';
import { launchImageLibrary } from 'react-native-image-picker';
import { FFmpegKit, ReturnCode } from '@wokcito/ffmpeg-kit-react-native';
import RNFS from 'react-native-fs';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { LineChart } from 'react-native-chart-kit';
import * as base64js from 'base64-js';

const LEFT_SHOULDER = 5;
const RIGHT_SHOULDER = 6;
const LEFT_ELBOW = 7;
const RIGHT_ELBOW = 8;
const LEFT_WRIST = 9;
const RIGHT_WRIST = 10;
const MIN_CONFIDENCE = 0.3;

function calculateAngle(
  a: { x: number; y: number },
  b: { x: number; y: number },
  c: { x: number; y: number },
): number {
  const radians =
    Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
  let angle = Math.abs((radians * 180) / Math.PI);
  if (angle > 180) angle = 360 - angle;
  return angle;
}

export default function VideoAnalysisScreen() {
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const [chartData, setChartData] = useState<any>(null);

  const plugin = useTensorflowModel(
    require('../assets/movenet_lightning.tflite'),
    [],
  );

  const actualModel = plugin.state === 'loaded' ? plugin.model : undefined;

  const pickVideo = async () => {
    const result = await launchImageLibrary({ mediaType: 'video' });
    if (result.assets && result.assets.length > 0) {
      const uri = result.assets[0].uri;
      if (uri) {
        processVideo(uri);
      }
    }
  };

  const processVideo = async (uri: string) => {
    if (!actualModel) return;
    setLoading(true);
    setChartData(null);
    setStatus('Extracting frames...');
    
    try {
      const tempDir = `${RNFS.CachesDirectoryPath}/frames`;
      const dirExists = await RNFS.exists(tempDir);
      if (dirExists) {
        await RNFS.unlink(tempDir);
      }
      await RNFS.mkdir(tempDir);

      const fps = 5;
      const ffmpegCommand = `-i "${uri}" -vf "fps=${fps},scale=192:192" -f image2 -pix_fmt rgb24 "${tempDir}/frame_%04d.raw"`;
      
      const session = await FFmpegKit.execute(ffmpegCommand);
      const returnCode = await session.getReturnCode();
      
      if (!ReturnCode.isSuccess(returnCode)) {
        setStatus('Failed to extract frames.');
        setLoading(false);
        return;
      }

      setStatus('Analyzing frames...');
      const files = await RNFS.readDir(tempDir);
      const rawFiles = files
        .filter(f => f.name.endsWith('.raw'))
        .sort((a, b) => a.name.localeCompare(b.name));
      
      const angles: number[] = [];
      const labels: string[] = [];
      
      for (let i = 0; i < rawFiles.length; i++) {
        const file = rawFiles[i];
        setStatus(`Analyzing frame ${i + 1}/${rawFiles.length}...`);
        
        const base64Str = await RNFS.readFile(file.path, 'base64');
        const buffer = base64js.toByteArray(base64Str);
        
        if (buffer.length === 110592) {
          const outputs = await actualModel.run([buffer.buffer]);
          const kp = new Float32Array(outputs[0] as ArrayBuffer);

          const rShoulder = { y: kp[RIGHT_SHOULDER * 3], x: kp[RIGHT_SHOULDER * 3 + 1], score: kp[RIGHT_SHOULDER * 3 + 2] };
          const rElbow = { y: kp[RIGHT_ELBOW * 3], x: kp[RIGHT_ELBOW * 3 + 1], score: kp[RIGHT_ELBOW * 3 + 2] };
          const rWrist = { y: kp[RIGHT_WRIST * 3], x: kp[RIGHT_WRIST * 3 + 1], score: kp[RIGHT_WRIST * 3 + 2] };
          const rScore = Math.min(rShoulder.score, rElbow.score, rWrist.score);
          
          const lShoulder = { y: kp[LEFT_SHOULDER * 3], x: kp[LEFT_SHOULDER * 3 + 1], score: kp[LEFT_SHOULDER * 3 + 2] };
          const lElbow = { y: kp[LEFT_ELBOW * 3], x: kp[LEFT_ELBOW * 3 + 1], score: kp[LEFT_ELBOW * 3 + 2] };
          const lWrist = { y: kp[LEFT_WRIST * 3], x: kp[LEFT_WRIST * 3 + 1], score: kp[LEFT_WRIST * 3 + 2] };
          const lScore = Math.min(lShoulder.score, lElbow.score, lWrist.score);

          const timeSec = (i / fps).toFixed(1);
          labels.push(timeSec);
          
          if (rScore >= MIN_CONFIDENCE && rScore >= lScore) {
            angles.push(calculateAngle(rShoulder, rElbow, rWrist));
          } else if (lScore >= MIN_CONFIDENCE) {
            angles.push(calculateAngle(lShoulder, lElbow, lWrist));
          } else {
            // Keep timeline continuous by using last known angle, or 0 if none yet
            angles.push(angles.length > 0 ? angles[angles.length - 1] : 0);
          }
        }
      }
      
      await RNFS.unlink(tempDir);
      
      if (angles.length > 0) {
        const step = Math.ceil(labels.length / 6);
        const filteredLabels = labels.map((l, idx) => (idx % step === 0 ? l : ''));
        
        setChartData({
          labels: filteredLabels,
          datasets: [
            {
              data: angles,
            },
          ],
        });
        setStatus('Analysis complete!');
      } else {
        setStatus('Could not detect elbow in the video.');
      }
    } catch (error) {
      console.error(error);
      setStatus('An error occurred during processing.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Video Angle Analysis</Text>
      <Button 
        title={!actualModel ? "Loading model..." : "Select Video"} 
        onPress={pickVideo} 
        disabled={loading || !actualModel} 
      />
      
      {loading && (
        <View style={styles.statusContainer}>
          <ActivityIndicator size="large" color="#0000ff" />
          <Text style={styles.statusText}>{status}</Text>
        </View>
      )}
      
      {!loading && chartData && (
        <View style={styles.chartContainer}>
          <Text style={styles.statusText}>{status}</Text>
          <ScrollView horizontal={true} showsHorizontalScrollIndicator={true} style={styles.scrollView}>
            <LineChart
              data={chartData}
              width={Math.max(Dimensions.get('window').width - 32, chartData.datasets[0].data.length * 80)}
              height={220}
              yAxisSuffix="°"
              chartConfig={{
                backgroundColor: '#ffffff',
                backgroundGradientFrom: '#ffffff',
                backgroundGradientTo: '#ffffff',
                decimalPlaces: 0,
                color: (opacity = 1) => `rgba(0, 0, 255, ${opacity})`,
                labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
                style: {
                  borderRadius: 16,
                },
                propsForDots: {
                  r: '3',
                  strokeWidth: '1',
                  stroke: '#ffa726',
                },
              }}
              bezier
              style={{
                marginVertical: 8,
                borderRadius: 16,
              }}
            />
          </ScrollView>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 16, backgroundColor: '#fff', alignItems: 'center' },
  title: { fontSize: 24, fontWeight: 'bold', marginBottom: 20, marginTop: 40 },
  statusContainer: { marginTop: 40, alignItems: 'center' },
  statusText: { marginTop: 10, fontSize: 16, color: '#333', textAlign: 'center', marginBottom: 10 },
  chartContainer: { marginTop: 40, width: '100%' },
  scrollView: { width: '100%' },
});
