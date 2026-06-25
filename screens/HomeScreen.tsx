import { View, Text, Button } from 'react-native'
import React from 'react'
import { useNavigation } from '@react-navigation/native'

const HomeScreen = () => {
    const navigation = useNavigation()
  return (
    <View>
      <Text>HomeScreen</Text>
      <Button title="Elbow Angle" onPress={() => navigation.navigate('Elbow')} />
    </View>
  )
}

export default HomeScreen