import {
  Camera,
  useCameraDevices,
  useCameraPermission,
} from 'react-native-vision-camera';
import { useTensorflowModel } from 'react-native-fast-tflite';
import { View, Text, Pressable } from 'react-native';
import { } from 'react-native-vision-camera-resizer'
import { useEffect } from 'react';

export default function App() {
  const devices = useCameraDevices();
  const device = devices.find(d => d.position === 'front');
  const { hasPermission, requestPermission } = useCameraPermission();

  const model = useTensorflowModel(require('./assets/movenet_lightning.tflite'), []);

  const logState = () => {
    alert("hi")
    const inputTensor = model.model?.inputs?.[0];
    console.log("hi")
    console.log('Full input tensor:', inputTensor);
    console.log('Input shape:', inputTensor?.shape);
  }
  

  if (!hasPermission) {
    return (
      <View>
        <Text onPress={requestPermission}>Grant Camera Permission</Text>
      </View>
    );
  }

  if (!device) {
    return <Text>No Camera Found</Text>;
  }

  // 👇 Step 4 check — just show model state, don't touch the camera frames yet
  if (model.state !== 'loaded') {
    return (
      <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
        <Text>Model state: {model.state}</Text>
        {model.state === 'error' && <Text>{String(model.error)}</Text>}
      </View>
    );
  }

  return (
    <View style={{ flex: 1 }}>
      <Camera style={{ flex: 1 }} device={device} isActive={true} />
      <View style={{ position: 'absolute', top: 50, left: 20 }}>
        <Pressable onPress={() => logState()}><Text style={{ color: 'lime', fontSize: 18 }}>Model loaded ✅</Text></Pressable>
        <Text>Model Size: {model.model?.inputs?.[0]?.shape?.[0]}x{model.model?.inputs?.[0]?.shape?.[1]}</Text>
      </View>
    </View>
  );
}