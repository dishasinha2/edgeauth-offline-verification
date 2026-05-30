import React from 'react';
import {
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  View,
} from 'react-native';

function App() {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />

      <View style={styles.logoContainer}>
        <Text style={styles.logo}>🛡️</Text>
      </View>

      <Text style={styles.title}>EdgeAuth</Text>

      <Text style={styles.subtitle}>
        Secure • Offline • Reliable
      </Text>

      <Text style={styles.loading}>
        Initializing...
      </Text>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0A0F14',
    justifyContent: 'center',
    alignItems: 'center',
  },

  logoContainer: {
    marginBottom: 20,
  },

  logo: {
    fontSize: 80,
  },

  title: {
    color: '#FFFFFF',
    fontSize: 42,
    fontWeight: '700',
  },

  subtitle: {
    color: '#8B949E',
    fontSize: 18,
    marginTop: 10,
  },

  loading: {
    color: '#00E676',
    fontSize: 16,
    marginTop: 80,
  },
});

export default App;