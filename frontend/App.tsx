import React, {useEffect, useState} from 'react';
import {
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
} from 'react-native';

function SplashScreen() {
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

      <Text style={styles.loading}>Initializing...</Text>
    </SafeAreaView>
  );
}

function LoginScreen() {
  return (
    <SafeAreaView style={styles.loginContainer}>
      <StatusBar barStyle="light-content" />

      <View style={styles.header}>
        <Text style={styles.smallLabel}>EDGEAUTH</Text>

        <Text style={styles.loginTitle}>
          Workforce Identity Verification
        </Text>

        <Text style={styles.loginSubtitle}>
          Secure employee authentication for offline environments.
        </Text>
      </View>

      <View style={styles.form}>
        <Text style={styles.inputLabel}>Employee ID</Text>
        <TextInput
          placeholder="Enter employee ID"
          placeholderTextColor="#6B7280"
          style={styles.input}
        />

        <Text style={styles.inputLabel}>Organization Code</Text>
        <TextInput
          placeholder="Enter organization code"
          placeholderTextColor="#6B7280"
          style={styles.input}
        />

        <TouchableOpacity style={styles.button}>
          <Text style={styles.buttonText}>
            Continue Verification
          </Text>
        </TouchableOpacity>

        <View style={styles.statusBox}>
          <View style={styles.statusDot} />
          <Text style={styles.statusText}>
            Offline Verification Enabled
          </Text>
        </View>
      </View>
    </SafeAreaView>
  );
}

function App() {
  const [showSplash, setShowSplash] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowSplash(false);
    }, 2200);

    return () => clearTimeout(timer);
  }, []);

  return showSplash ? <SplashScreen /> : <LoginScreen />;
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

  loginContainer: {
    flex: 1,
    backgroundColor: '#0A0F14',
    paddingHorizontal: 28,
    justifyContent: 'center',
  },

  header: {
    marginBottom: 50,
  },

  smallLabel: {
    color: '#00E676',
    letterSpacing: 2,
    fontWeight: '700',
    fontSize: 12,
    marginBottom: 16,
  },

  loginTitle: {
    color: '#FFFFFF',
    fontSize: 34,
    fontWeight: '700',
    lineHeight: 42,
  },

  loginSubtitle: {
    color: '#8B949E',
    fontSize: 16,
    marginTop: 12,
    lineHeight: 24,
  },

  form: {
    marginTop: 10,
  },

  inputLabel: {
    color: '#FFFFFF',
    marginBottom: 8,
    fontSize: 14,
    fontWeight: '600',
  },

  input: {
    backgroundColor: '#141B22',
    borderRadius: 14,
    paddingHorizontal: 18,
    paddingVertical: 16,
    color: '#FFFFFF',
    marginBottom: 22,
    fontSize: 15,
  },

  button: {
    backgroundColor: '#00E676',
    borderRadius: 14,
    paddingVertical: 18,
    alignItems: 'center',
    marginTop: 10,
  },

  buttonText: {
    color: '#000000',
    fontWeight: '700',
    fontSize: 16,
  },

  statusBox: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 28,
    justifyContent: 'center',
  },

  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 10,
    backgroundColor: '#00E676',
    marginRight: 8,
  },

  statusText: {
    color: '#8B949E',
    fontSize: 14,
  },
});

export default App;