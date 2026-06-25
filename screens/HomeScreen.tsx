import { View, Text, Button } from 'react-native'
import React from 'react'
import { useNavigation } from '@react-navigation/native'

const HomeScreen = () => {
    const navigation = useNavigation()
  return (
    <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', gap: 16 }}>
      <Text style={{ fontSize: 24, fontWeight: 'bold' }}>FormScore</Text>
      <Button title="Real-time Elbow Angle" onPress={() => navigation.navigate('Elbow')} />
      <Button title="Video Analysis" onPress={() => navigation.navigate('VideoAnalysis')} />
    </View>
  )
}

export default HomeScreen