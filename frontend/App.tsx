import React, {useEffect, useState} from 'react';
import {launchCamera} from 'react-native-image-picker';
import {
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  Alert,
} from 'react-native';

function SplashScreen() {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />

      <Text style={styles.logo}>🛡️</Text>

      <Text style={styles.title}>EdgeAuth</Text>

      <Text style={styles.subtitle}>
        Secure • Offline • Reliable
      </Text>

      <Text style={styles.loading}>Initializing...</Text>
    </SafeAreaView>
  );
}

function VerificationScreen() {
  const openCamera = async () => {
    try {
      const response = await launchCamera({
  mediaType: 'photo',
  includeBase64: true,
  cameraType: 'front',
  saveToPhotos: false,
});

      console.log('CAMERA RESPONSE:', response);

      if (response.didCancel) {
        console.log('User cancelled camera');
        return;
      }

      if (response.errorCode) {
        Alert.alert(
          'Camera Error',
          response.errorMessage || 'Unknown camera error',
        );
        return;
      }

      Alert.alert(
        'Success',
        'Photo captured successfully!',
      );
    } catch (error) {
      console.log('CAMERA ERROR:', error);
      Alert.alert(
        'Error',
        String(error),
      );
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" />

      <TouchableOpacity
        style={{
          backgroundColor: '#00E676',
          paddingHorizontal: 40,
          paddingVertical: 18,
          borderRadius: 18,
        }}
        onPress={openCamera}>
        <Text
          style={{
            color: '#000000',
            fontSize: 18,
            fontWeight: '700',
          }}>
          Open Camera
        </Text>
      </TouchableOpacity>
    </SafeAreaView>
  );
}
function LoginScreen({onSuccess}: any) {
  const [employeeId, setEmployeeId] = useState('');
  const [organizationCode, setOrganizationCode] = useState('');

  const handleLogin = () => {
    const validUsers = [
      {employeeId: 'EMP-A001', orgCode: 'ORG-ALPHA'},
      {employeeId: 'EMP-A002', orgCode: 'ORG-ALPHA'},
      {employeeId: 'EMP-B001', orgCode: 'ORG-BETA'},
      {employeeId: 'EMP-B002', orgCode: 'ORG-BETA'},
      {employeeId: 'EMP-G001', orgCode: 'ORG-GAMMA'},
      {employeeId: 'EMP-G002', orgCode: 'ORG-GAMMA'},
    ];

    const isValid = validUsers.some(
      user =>
        user.employeeId.toUpperCase() === employeeId.trim().toUpperCase() &&
        user.orgCode.toUpperCase() === organizationCode.trim().toUpperCase(),
    );

    if (isValid) {
      onSuccess();
    } else {
      Alert.alert(
        'Invalid Credentials',
        'Employee ID or Organization Code is incorrect.',
      );
    }
  };

  return (
    <SafeAreaView style={styles.loginContainer}>
      <StatusBar barStyle="light-content" />

      <View style={styles.header}>
        <Text style={styles.smallLabel}>EDGEAUTH</Text>

        <Text style={styles.loginTitle}>
          Workforce Identity
        </Text>

        <Text style={styles.loginTitle}>
          Verification
        </Text>

        <Text style={styles.loginSubtitle}>
          Secure employee authentication for offline environments.
        </Text>
      </View>

      <View style={styles.form}>
        <Text style={styles.inputLabel}>Employee ID</Text>

        <TextInput
          placeholder="Enter employee ID"
          placeholderTextColor="#667085"
          style={styles.input}
          value={employeeId}
          onChangeText={setEmployeeId}
          autoCapitalize="characters"
        />

        <Text style={styles.inputLabel}>Organization Code</Text>

        <TextInput
          placeholder="Enter organization code"
          placeholderTextColor="#667085"
          style={styles.input}
          value={organizationCode}
          onChangeText={setOrganizationCode}
          autoCapitalize="characters"
        />

        <TouchableOpacity
          style={styles.button}
          onPress={handleLogin}>
          <Text style={styles.buttonText}>
            Continue Verification
          </Text>
        </TouchableOpacity>
      </View>
    </SafeAreaView>
  );
}

function App() {
  const [showSplash, setShowSplash] = useState(true);
  const [verified, setVerified] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setShowSplash(false);
    }, 2200);

    return () => clearTimeout(timer);
  }, []);

  if (showSplash) {
    return <SplashScreen />;
  }

  if (verified) {
    return <VerificationScreen />;
  }

  return (
    <LoginScreen
      onSuccess={() => setVerified(true)}
    />
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#071018',
    justifyContent: 'center',
    alignItems: 'center',
  },

  logo: {
    fontSize: 82,
    marginBottom: 20,
  },

  title: {
    color: '#FFFFFF',
    fontSize: 44,
    fontWeight: '700',
  },

  subtitle: {
    color: '#8B949E',
    fontSize: 17,
    marginTop: 10,
  },

  loading: {
    color: '#00E676',
    fontSize: 16,
    marginTop: 90,
  },

  loginContainer: {
    flex: 1,
    backgroundColor: '#071018',
    paddingHorizontal: 30,
    justifyContent: 'center',
  },

  header: {
    marginBottom: 55,
  },

  smallLabel: {
    color: '#00E676',
    fontSize: 12,
    fontWeight: '700',
    letterSpacing: 3,
    marginBottom: 20,
  },

  loginTitle: {
    color: '#FFFFFF',
    fontSize: 30,
    fontWeight: '700',
    lineHeight: 36,
  },

  loginSubtitle: {
    color: '#7C8592',
    fontSize: 16,
    lineHeight: 24,
    marginTop: 18,
  },

  form: {
    marginTop: 10,
  },

  inputLabel: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
    marginBottom: 10,
  },

  input: {
    backgroundColor: '#111A23',
    borderRadius: 18,
    paddingHorizontal: 18,
    paddingVertical: 18,
    color: '#FFFFFF',
    fontSize: 15,
    marginBottom: 24,
  },

  button: {
    backgroundColor: '#00E676',
    borderRadius: 18,
    paddingVertical: 18,
    alignItems: 'center',
    marginTop: 8,
  },

  buttonText: {
    color: '#000000',
    fontSize: 17,
    fontWeight: '700',
  },

  successContainer: {
    flex: 1,
    backgroundColor: '#071018',
    justifyContent: 'center',
    paddingHorizontal: 24,
  },

  successBrand: {
    color: '#00E676',
    textAlign: 'center',
    letterSpacing: 4,
    fontWeight: '700',
    marginBottom: 24,
  },

  successCard: {
    backgroundColor: '#111A23',
    borderRadius: 28,
    padding: 28,
  },

  successTitle: {
    color: '#FFFFFF',
    textAlign: 'center',
    fontSize: 30,
    fontWeight: '700',
  },

  successSubtitle: {
    color: '#8B949E',
    textAlign: 'center',
    marginTop: 10,
  },

  scoreCircle: {
    width: 160,
    height: 160,
    borderRadius: 80,
    borderWidth: 5,
    borderColor: '#00E676',
    justifyContent: 'center',
    alignItems: 'center',
    alignSelf: 'center',
    marginTop: 28,
  },

  scoreText: {
    color: '#FFFFFF',
    fontSize: 34,
    fontWeight: '700',
  },

  verifiedText: {
    color: '#00E676',
    textAlign: 'center',
    fontSize: 24,
    fontWeight: '700',
    marginTop: 24,
  },

  divider: {
    height: 1,
    backgroundColor: '#24303B',
    marginVertical: 24,
  },

  infoText: {
    color: '#FFFFFF',
    fontSize: 16,
    marginBottom: 10,
  },

  proceedButton: {
    backgroundColor: '#00E676',
    borderRadius: 18,
    paddingVertical: 18,
    marginTop: 24,
  },

  proceedButtonText: {
    color: '#000000',
    textAlign: 'center',
    fontSize: 16,
    fontWeight: '700',
  },
});

export default App;