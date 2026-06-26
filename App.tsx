import * as React from 'react';
import { View, Text } from 'react-native';
import { createStaticNavigation } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import HomeScreen from './screens/HomeScreen';
import ElbowAngleScreen from './screens/ElbowAngle';
import VideoAnalysisScreen from './screens/VideoAnalysisScreen';
import BicepCurlFormScreen from './screens/FormScoreScreen';

const RootStack = createNativeStackNavigator({
  initialRouteName: 'Home',
  screens: {
    Home: HomeScreen,
    Elbow: ElbowAngleScreen,
    VideoAnalysis: VideoAnalysisScreen,
    FormScore : BicepCurlFormScreen
  },
});

const Navigation = createStaticNavigation(RootStack);

export default function App() {
  return <Navigation />;
}