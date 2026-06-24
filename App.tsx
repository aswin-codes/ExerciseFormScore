import {
  Camera,
  useCameraDevices,
  useCameraPermission
} from 'react-native-vision-camera'
import { View, Text } from 'react-native'

export default function App() {
  const devices = useCameraDevices()
  const device = devices.find(d => d.position === 'front')

  // ✅ new hook-based permission API
  const { hasPermission, requestPermission } = useCameraPermission()

  if (!hasPermission) {
    return (
      <View>
        <Text onPress={requestPermission}>
          Grant Camera Permission
        </Text>
      </View>
    )
  }

  if (!device) {
    return <Text>No Camera Found</Text>
  }

  return (
    <Camera
      style={{ flex: 1 }}
      device={device}
      isActive={true}
    />
  )
}