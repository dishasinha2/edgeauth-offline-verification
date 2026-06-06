import React, {
  useState,
  useRef,
  useEffect,
  useContext,
  createContext,
} from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  TextInput,
  ScrollView,
  Animated,
  Dimensions,
  StatusBar,
  SafeAreaView,
  FlatList,
  Platform,
} from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';

const { width: SW, height: SH } = Dimensions.get('window');


const T = {
  bg:      '#0a0f0a',
  bg2:     '#0f160f',
  bg3:     '#141c14',
  card:    '#161e16',
  card2:   '#1c261c',
  border:  '#2a3a2a',
  accent:  '#00e676',
  accent2: '#69f0ae',
  accent3: '#1b5e20',
  danger:  '#ff5252',
  warn:    '#ffab40',
  text:    '#e8f5e8',
  text2:   '#9eb89e',
  text3:   '#5a7a5a',
  white:   '#ffffff',
};


const AuthContext = createContext<{
  role: string;
  setRole: (r: string) => void;
}>({ role: 'Admin', setRole: () => {} });


const RootStack  = createStackNavigator();
const AuthStack  = createStackNavigator();
const Tab        = createBottomTabNavigator();
const MainStack  = createStackNavigator();


const PrimaryBtn = ({
  label,
  onPress,
  style,
}: {
  label: string;
  onPress: () => void;
  style?: object;
}) => (
  <TouchableOpacity onPress={onPress} activeOpacity={0.85} style={[styles.primaryBtn, style]}>
    <Text style={styles.primaryBtnText}>{label}</Text>
  </TouchableOpacity>
);


const SecondaryBtn = ({
  label,
  onPress,
  style,
}: {
  label: string;
  onPress: () => void;
  style?: object;
}) => (
  <TouchableOpacity onPress={onPress} activeOpacity={0.8} style={[styles.secondaryBtn, style]}>
    <Text style={styles.secondaryBtnText}>{label}</Text>
  </TouchableOpacity>
);


const InputField = ({
  icon,
  placeholder,
  secureTextEntry,
}: {
  icon: string;
  placeholder: string;
  secureTextEntry?: boolean;
}) => (
  <View style={styles.inputWrap}>
    <Text style={styles.inputIcon}>{icon}</Text>
    <TextInput
      style={styles.input}
      placeholder={placeholder}
      placeholderTextColor={T.text3}
      secureTextEntry={secureTextEntry}
      autoCapitalize="none"
    />
  </View>
);


const OfflineBanner = ({ pending = 12 }: { pending?: number }) => {
  const blink = useRef(new Animated.Value(1)).current;
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(blink, { toValue: 0.2, duration: 700, useNativeDriver: true }),
        Animated.timing(blink, { toValue: 1,   duration: 700, useNativeDriver: true }),
      ]),
    ).start();
  }, []);
  return (
    <View style={styles.offlineBanner}>
      <Animated.View style={[styles.offlineDot, { opacity: blink }]} />
      <Text style={styles.offlineText}>OFFLINE MODE ACTIVE</Text>
      <Text style={styles.offlineCount}>{pending} pending</Text>
    </View>
  );
};


const ShieldLogo = ({ size = 60 }: { size?: number }) => (
  <View
    style={[
      styles.shieldWrap,
      { width: size, height: size * 1.12, borderRadius: size * 0.15 },
    ]}>
    <Text style={{ fontSize: size * 0.42 }}>🛡️</Text>
  </View>
);


// 1. SPLASH SCREEN

const SplashScreen = ({ navigation }: any) => {
  const fadeAnim  = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.8)).current;
  const spinAnim  = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    
    Animated.parallel([
      Animated.timing(fadeAnim,  { toValue: 1, duration: 900, useNativeDriver: true }),
      Animated.spring(scaleAnim, { toValue: 1, friction: 5, tension: 60, useNativeDriver: true }),
    ]).start();

    
    Animated.loop(
      Animated.timing(spinAnim, { toValue: 1, duration: 900, useNativeDriver: true }),
    ).start();

    
    const t = setTimeout(() => navigation.replace('Login'), 2800);
    return () => clearTimeout(t);
  }, []);

  const spin = spinAnim.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });

  return (
    <View style={styles.splashContainer}>
      <StatusBar barStyle="light-content" backgroundColor={T.bg} />

      
      <View style={StyleSheet.absoluteFillObject} pointerEvents="none">
        {Array.from({ length: 20 }).map((_, i) => (
          <View
            key={`h${i}`}
            style={[styles.gridLine, { top: i * 42, width: '100%', height: 1 }]}
          />
        ))}
        {Array.from({ length: 12 }).map((_, i) => (
          <View
            key={`v${i}`}
            style={[styles.gridLine, { left: i * 42, height: '100%', width: 1 }]}
          />
        ))}
      </View>

      <Animated.View style={{ opacity: fadeAnim, transform: [{ scale: scaleAnim }], alignItems: 'center' }}>
        {/* Rings */}
        <View style={[styles.ring, { width: 140, height: 140, top: -20, left: -20, position: 'absolute' }]} />
        <View style={[styles.ring, { width: 180, height: 180, top: -40, left: -40, position: 'absolute', opacity: 0.4 }]} />

        <ShieldLogo size={90} />

        <Text style={styles.splashAppName}>
          Edge<Text style={{ color: T.accent }}>Auth</Text>
        </Text>
        <Text style={styles.splashTagline}>SECURE · OFFLINE · RELIABLE</Text>
        <Text style={styles.splashSub}>Offline Workforce Verification Platform</Text>
      </Animated.View>

      
      <Animated.View style={[styles.initRow, { opacity: fadeAnim }]}>
        <Animated.View style={[styles.spinnerCircle, { transform: [{ rotate: spin }] }]} />
        <Text style={styles.initText}>Initializing secure environment...</Text>
      </Animated.View>

      <Text style={styles.versionText}>v1.0.0</Text>
    </View>
  );
};

// 2. LOGIN SCREEN

const LoginScreen = ({ navigation }: any) => {
  const [selectedRole, setSelectedRole] = useState('Admin');
  const { setRole } = useContext(AuthContext);
  const fadeAnim = useRef(new Animated.Value(0)).current;
  const slideAnim = useRef(new Animated.Value(30)).current;

  useEffect(() => {
    Animated.parallel([
      Animated.timing(fadeAnim,  { toValue: 1, duration: 600, useNativeDriver: true }),
      Animated.timing(slideAnim, { toValue: 0, duration: 600, useNativeDriver: true }),
    ]).start();
  }, []);

  const roles = [
    { label: 'Admin',       icon: '🛡️' },
    { label: 'Employee',    icon: '👤' },
    { label: 'Super Admin', icon: '👑' },
  ];

  const handleLogin = () => {
    setRole(selectedRole);
    navigation.replace('MainApp');
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: T.bg }}>
      <StatusBar barStyle="light-content" backgroundColor={T.bg} />
      <ScrollView contentContainerStyle={styles.loginScroll} showsVerticalScrollIndicator={false}>
        <Animated.View style={{ opacity: fadeAnim, transform: [{ translateY: slideAnim }] }}>
          <ShieldLogo size={52} />
          <Text style={styles.loginTitle}>Welcome Back!</Text>
          <Text style={styles.loginSub}>Sign in to continue</Text>

         
          <View style={styles.roleRow}>
            {roles.map(r => (
              <TouchableOpacity
                key={r.label}
                style={[styles.roleTab, selectedRole === r.label && styles.roleTabActive]}
                onPress={() => setSelectedRole(r.label)}
                activeOpacity={0.8}>
                <Text style={{ fontSize: 22 }}>{r.icon}</Text>
                <Text style={[styles.roleTabLabel, selectedRole === r.label && { color: T.accent }]}>
                  {r.label}
                </Text>
              </TouchableOpacity>
            ))}
          </View>

          
         <Text style={styles.inputLabel}>EMAIL / USERNAME</Text>
          <InputField icon="" placeholder="Enter email or username" />

          <Text style={[styles.inputLabel, { marginTop: 14 }]}>PASSWORD</Text>
          <InputField icon="" placeholder="Enter password" secureTextEntry />

          <TouchableOpacity style={{ alignSelf: 'flex-end', marginTop: 8, marginBottom: 24 }}>
            <Text style={{ color: T.accent, fontSize: 12, fontWeight: '600' }}>Forgot Password?</Text>
          </TouchableOpacity>

          <PrimaryBtn label="Login" onPress={handleLogin} />

         
          <View style={styles.dividerRow}>
            <View style={styles.dividerLine} />
            <Text style={styles.dividerText}>or</Text>
            <View style={styles.dividerLine} />
          </View>

         
          <TouchableOpacity style={styles.qrBtn} onPress={handleLogin} activeOpacity={0.8}>
            <Text style={{ fontSize: 18 }}>📱</Text>
            <Text style={styles.qrBtnText}>Login with QR Code</Text>
          </TouchableOpacity>

          <TouchableOpacity style={{ marginTop: 20, alignSelf: 'center' }}>
            <Text style={{ color: T.text3, fontSize: 13 }}>
              Don't have an account?{' '}
              <Text style={{ color: T.accent, fontWeight: '700' }}>Register</Text>
            </Text>
          </TouchableOpacity>
        </Animated.View>
      </ScrollView>
    </SafeAreaView>
  );
};


// 3. DASHBOARD SCREEN

const DashboardScreen = ({ navigation }: any) => {
  const fadeAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.timing(fadeAnim, { toValue: 1, duration: 500, useNativeDriver: true }).start();
  }, []);

  const quickActions = [
    { icon: '➕', label: 'Add Employee',   screen: 'Enroll' },
    { icon: '👤', label: 'Verify Identity', screen: 'Verify' },
    { icon: '📋', label: 'View Logs',       screen: 'Logs'   },
    { icon: '📊', label: 'Reports',         screen: 'Reports'},
    { icon: '☁️', label: 'Pending Sync',    screen: 'Sync', badge: '12' },
    { icon: '⚙️', label: 'Settings',        screen: 'Profile'},
  ];

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: T.bg }}>
      <StatusBar barStyle="light-content" backgroundColor={T.bg} />
      <OfflineBanner />
      <ScrollView showsVerticalScrollIndicator={false}>
        <Animated.View style={{ opacity: fadeAnim }}>
          {/* Header */}
          <View style={styles.dashHeader}>
            <View>
              <Text style={styles.dashGreeting}>Hello,</Text>
              <Text style={styles.dashName}>Admin</Text>
              <Text style={styles.dashOrg}>Bennett University</Text>
            </View>
            <View style={{ flexDirection: 'row', gap: 10 }}>
              <View style={styles.iconBtn}>
                <Text style={{ fontSize: 16 }}>🔔</Text>
                <View style={styles.notifBadge}><Text style={{ color: '#fff', fontSize: 8, fontWeight: '800' }}>3</Text></View>
              </View>
              <View style={styles.iconBtn}><Text style={{ fontSize: 16 }}>☰</Text></View>
            </View>
          </View>

          
          <View style={styles.statCard}>
            <Text style={styles.statLabel}>👥  Total Employees</Text>
            <Text style={styles.statNum}>128</Text>
            <Text style={styles.statSub}>↑ +4 active this month</Text>
           
            <View style={styles.sparklineWrap}>
              {[40, 55, 48, 70, 62, 50, 80].map((h, i) => (
                <View
                  key={i}
                  style={[
                    styles.sparkBar,
                    { height: h * 0.5, opacity: 0.4 + i * 0.08 },
                  ]}
                />
              ))}
            </View>
          </View>

         
          <Text style={[styles.sectionTitle, { marginHorizontal: 20 }]}>QUICK ACTIONS</Text>
          <View style={styles.quickGrid}>
            {quickActions.map(a => (
              <TouchableOpacity
                key={a.label}
                style={styles.quickItem}
                onPress={() => navigation.navigate(a.screen)}
                activeOpacity={0.8}>
                <Text style={{ fontSize: 24 }}>{a.icon}</Text>
                <Text style={styles.quickLabel}>{a.label}</Text>
                {a.badge && (
                  <View style={styles.quickBadge}>
                    <Text style={{ color: '#fff', fontSize: 9, fontWeight: '800' }}>{a.badge}</Text>
                  </View>
                )}
              </TouchableOpacity>
            ))}
          </View>

          
          <Text style={[styles.sectionTitle, { marginHorizontal: 20 }]}>SYSTEM STATUS</Text>
          <View style={styles.statusCard}>
            {[
              { k: 'Mode',            v: '● Offline',    vc: T.danger },
              { k: 'Pending Records', v: '12',           vc: T.warn   },
              { k: 'Last Sync',       v: '2 days ago',   vc: T.text3  },
              { k: 'Model Status',    v: '● Ready',      vc: T.accent },
            ].map((row, i) => (
              <View
                key={i}
                style={[
                  styles.statusRow,
                  i === 3 && { borderBottomWidth: 0 },
                ]}>
                <Text style={styles.statusKey}>{row.k}</Text>
                <Text style={[styles.statusVal, { color: row.vc }]}>{row.v}</Text>
              </View>
            ))}
          </View>

          <View style={{ height: 20 }} />
        </Animated.View>
      </ScrollView>
    </SafeAreaView>
  );
};


// 4. VERIFY SCREEN  

const challenges = [
  { label: 'Blink Twice',      hint: 'Please blink your eyes twice' },
  { label: 'Turn Head Left',   hint: 'Please turn your head to the left' },
  { label: 'Smile',            hint: 'Please smile naturally' },
];

const VerifyScreen = ({ navigation }: any) => {
  const [step, setStep] = useState(0);
  const scanAnim  = useRef(new Animated.Value(0)).current;
  const glowAnim  = useRef(new Animated.Value(0.4)).current;

  useEffect(() => {
    
    Animated.loop(
      Animated.sequence([
        Animated.timing(scanAnim, { toValue: 1, duration: 2000, useNativeDriver: false }),
        Animated.timing(scanAnim, { toValue: 0, duration: 0,    useNativeDriver: false }),
      ]),
    ).start();

    
    Animated.loop(
      Animated.sequence([
        Animated.timing(glowAnim, { toValue: 1,   duration: 1000, useNativeDriver: true }),
        Animated.timing(glowAnim, { toValue: 0.4, duration: 1000, useNativeDriver: true }),
      ]),
    ).start();
  }, []);

  const scanTop = scanAnim.interpolate({
    inputRange: [0, 1],
    outputRange: [20, 220],
  });

  const handleNext = () => {
    if (step < challenges.length - 1) {
      setStep(s => s + 1);
    } else {
      navigation.navigate('Result');
    }
  };

  return (
    <View style={{ flex: 1, backgroundColor: '#000' }}>
      <StatusBar barStyle="light-content" backgroundColor="#000" />

      
      <View style={StyleSheet.absoluteFillObject}>
        <View style={styles.cameraBg} />
        {/* Scanline overlay */}
        {Array.from({ length: 60 }).map((_, i) => (
          <View
            key={i}
            style={[
              styles.scanStripe,
              { top: i * 14 },
            ]}
          />
        ))}
      </View>

      
      <SafeAreaView>
        <View style={styles.verifyTopBar}>
          <TouchableOpacity
            style={styles.verifyBackBtn}
            onPress={() => navigation.goBack()}>
            <Text style={{ color: T.white, fontSize: 18 }}>←</Text>
          </TouchableOpacity>
          <Text style={styles.verifyTitle}>Verify Identity</Text>
          <TouchableOpacity style={styles.flashBtn}>
            <Text style={{ fontSize: 18 }}>⚡</Text>
          </TouchableOpacity>
        </View>
        <Text style={styles.faceGuideText}>Position your face in the frame</Text>
      </SafeAreaView>

      
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
        <Animated.View style={[styles.ovalOuter, { opacity: glowAnim }]}>
          <View style={styles.ovalInner}>
            {/* Face placeholder */}
            <View style={styles.facePlaceholder}>
              <Text style={{ fontSize: 80 }}>👤</Text>
            </View>

            
            <Animated.View style={[styles.scanLineAnim, { top: scanTop }]} />

            
            <View style={[styles.corner, styles.cornerTL]} />
            <View style={[styles.corner, styles.cornerTR]} />
            <View style={[styles.corner, styles.cornerBL]} />
            <View style={[styles.corner, styles.cornerBR]} />
          </View>
        </Animated.View>
      </View>

     
      <View style={styles.livenessPanel}>
        <Text style={styles.challengeLabel}>
          Challenge · Step {step + 1} of {challenges.length}
        </Text>
        <Text style={styles.challengeText}>{challenges[step].label}</Text>

        
        <View style={{ flexDirection: 'row', gap: 8, marginBottom: 8 }}>
          {challenges.map((_, i) => (
            <View
              key={i}
              style={[
                styles.chDot,
                i < step  && styles.chDotDone,
                i === step && styles.chDotActive,
              ]}
            />
          ))}
        </View>

        <Text style={styles.challengeHint}>{challenges[step].hint}</Text>

        <View style={{ flexDirection: 'row', gap: 12, marginTop: 16 }}>
          <SecondaryBtn
            label="Cancel"
            onPress={() => navigation.goBack()}
            style={{ flex: 1 }}
          />
          <PrimaryBtn
            label={step < challenges.length - 1 ? 'Next →' : 'Confirm ✓'}
            onPress={handleNext}
            style={{ flex: 2 }}
          />
        </View>
      </View>
    </View>
  );
};

// 5. RESULT SCREEN

const ResultScreen = ({ navigation }: any) => {
  const scaleAnim = useRef(new Animated.Value(0)).current;
  const barAnim   = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    Animated.sequence([
      Animated.spring(scaleAnim, { toValue: 1, friction: 4, tension: 60, useNativeDriver: true }),
      Animated.timing(barAnim,   { toValue: 0.98, duration: 800, useNativeDriver: false }),
    ]).start();
  }, []);

  const barWidth = barAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%'],
  });

  const meta = [
    { k: 'Timestamp',    v: '29 May 2026 · 09:41 AM' },
    { k: 'Liveness',     v: '✓ Passed',     vc: T.accent },
    { k: 'Sync Status',  v: '⏳ Pending',   vc: T.warn   },
    { k: 'Device',       v: 'EDGEAUTH-ANDROID-23' },
  ];

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: T.bg }}>
      <StatusBar barStyle="light-content" backgroundColor={T.bg} />
      <View style={styles.resultTopBar}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.navigate('Dashboard')}>
          <Text style={{ color: T.text, fontSize: 18 }}>←</Text>
        </TouchableOpacity>
        <Text style={styles.screenTitle}>Verification Result</Text>
        <View style={{ width: 36 }} />
      </View>

      <ScrollView contentContainerStyle={{ padding: 20 }} showsVerticalScrollIndicator={false}>
        
        <Animated.View style={[styles.resultIconWrap, { transform: [{ scale: scaleAnim }] }]}>
          <Text style={{ fontSize: 52 }}>✅</Text>
        </Animated.View>
        <Text style={styles.resultTitle}>Verified Successfully!</Text>
        <Text style={styles.resultSub}>Identity confirmed · Offline Mode</Text>

        
        <View style={styles.resultCard}>
          <View style={{ flexDirection: 'row', alignItems: 'center', gap: 14, marginBottom: 16 }}>
            <View style={styles.resultAvatar}><Text style={{ fontSize: 28 }}>👨🏽‍💼</Text></View>
            <View>
              <Text style={styles.resultPersonName}>Aarav Sharma</Text>
              <Text style={styles.resultPersonId}>EMP001 · Engineering</Text>
            </View>
          </View>

         
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 }}>
            <Text style={styles.statusKey}>Confidence Score</Text>
            <Text style={[styles.statNum, { fontSize: 22 }]}>98%</Text>
          </View>
          <View style={styles.confidenceTrack}>
            <Animated.View style={[styles.confidenceFill, { width: barWidth }]} />
          </View>

          
          {meta.map((m, i) => (
            <View
              key={i}
              style={[
                styles.metaRow,
                i === 0 && { marginTop: 12 },
              ]}>
              <Text style={styles.statusKey}>{m.k}</Text>
              <Text style={[styles.metaVal, m.vc ? { color: m.vc } : {}]}>{m.v}</Text>
            </View>
          ))}
        </View>

        <PrimaryBtn
          label="Back to Dashboard"
          onPress={() => navigation.navigate('Dashboard')}
          style={{ marginTop: 8 }}
        />
        <SecondaryBtn
          label="View All Logs"
          onPress={() => navigation.navigate('Logs')}
          style={{ marginTop: 10 }}
        />
      </ScrollView>
    </SafeAreaView>
  );
};


// 6. EMPLOYEES SCREEN

const employees = [
  { id: 'EMP001', name: 'Aarav Sharma',  dept: 'Engineering', active: true,  icon: '👨🏽‍💼' },
  { id: 'EMP002', name: 'Diya Patel',    dept: 'Design',      active: true,  icon: '👩🏽‍💼' },
  { id: 'EMP003', name: 'Rohan Verma',   dept: 'Operations',  active: true,  icon: '👨🏾‍💼' },
  { id: 'EMP004', name: 'Sneha Iyer',    dept: 'HR',          active: true,  icon: '👩🏻‍💼' },
  { id: 'EMP005', name: 'Karan Singh',   dept: 'Security',    active: false, icon: '👨🏽‍💼' },
  { id: 'EMP006', name: 'Meera Nair',    dept: 'Finance',     active: true,  icon: '👩🏻‍💼' },
];

const EmployeesScreen = ({ navigation }: any) => {
  const [query, setQuery] = useState('');
  const filtered = employees.filter(
    e => e.name.toLowerCase().includes(query.toLowerCase()) || e.id.includes(query),
  );

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: T.bg }}>
      <StatusBar barStyle="light-content" backgroundColor={T.bg} />
      <View style={styles.topHeader}>
        <Text style={styles.screenTitle}>Employees</Text>
        <View style={{ flexDirection: 'row', gap: 8 }}>
          <View style={styles.iconBtn}><Text style={{ fontSize: 16 }}>🔍</Text></View>
          <View style={styles.iconBtn}><Text style={{ fontSize: 16 }}>⚙️</Text></View>
        </View>
      </View>

      <View style={{ paddingHorizontal: 20 }}>
        {/* Search */}
        <View style={styles.searchBar}>
          <Text style={{ fontSize: 16, color: T.text3 }}>🔍</Text>
          <TextInput
            style={styles.searchInput}
            placeholder="Search employees..."
            placeholderTextColor={T.text3}
            value={query}
            onChangeText={setQuery}
          />
        </View>

        <TouchableOpacity
          style={styles.addEmpBtn}
          onPress={() => navigation.navigate('Enroll')}
          activeOpacity={0.8}>
          <Text style={{ color: T.accent, fontSize: 16 }}>➕</Text>
          <Text style={styles.addEmpText}>Add Employee</Text>
        </TouchableOpacity>

        <Text style={styles.sectionTitle}>ALL EMPLOYEES · {filtered.length}</Text>
      </View>

      <FlatList
        data={filtered}
        keyExtractor={e => e.id}
        contentContainerStyle={{ paddingHorizontal: 20, paddingBottom: 20 }}
        showsVerticalScrollIndicator={false}
        renderItem={({ item: e }) => (
          <TouchableOpacity style={styles.empItem} activeOpacity={0.8}>
            <View style={styles.empAvatar}>
              <Text style={{ fontSize: 22 }}>{e.icon}</Text>
            </View>
            <View style={{ flex: 1 }}>
              <Text style={styles.empName}>{e.name}</Text>
              <Text style={styles.empId}>{e.id} · {e.dept}</Text>
            </View>
            <View
              style={[
                styles.empStatusBadge,
                { backgroundColor: e.active ? 'rgba(0,230,118,0.1)' : 'rgba(255,82,82,0.1)' },
              ]}>
              <Text
                style={{
                  fontSize: 10,
                  fontWeight: '700',
                  color: e.active ? T.accent : T.danger,
                }}>
                {e.active ? 'Active' : 'Inactive'}
              </Text>
            </View>
          </TouchableOpacity>
        )}
      />
    </SafeAreaView>
  );
};


// 7. ENROLL SCREEN 

const EnrollScreen = ({ navigation }: any) => {
  const [step, setStep] = useState(0);
  const stepTitles = ['Details', 'Capture', 'Review'];

  const goNext = () => {
    if (step < 2) setStep(s => s + 1);
    else navigation.navigate('Employees');
  };
  const goBack = () => {
    if (step > 0) setStep(s => s - 1);
    else navigation.goBack();
  };

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: T.bg }}>
      <StatusBar barStyle="light-content" backgroundColor={T.bg} />
      <View style={styles.topHeader}>
        <TouchableOpacity style={styles.backBtn} onPress={goBack}>
          <Text style={{ color: T.text, fontSize: 18 }}>←</Text>
        </TouchableOpacity>
        <Text style={styles.screenTitle}>Add Employee</Text>
        <View style={{ width: 36 }} />
      </View>

      
      <View style={styles.stepsRow}>
        {stepTitles.map((t, i) => (
          <React.Fragment key={t}>
            <View
              style={[
                styles.stepBubble,
                i < step  && styles.stepBubbleDone,
                i === step && styles.stepBubbleActive,
              ]}>
              <Text
                style={{
                  fontSize: 11,
                  fontWeight: '800',
                  color: i < step ? T.bg : i === step ? T.accent : T.text3,
                }}>
                {i < step ? '✓' : i + 1}
              </Text>
            </View>
            {i < 2 && (
              <View style={[styles.stepLine, i < step && { backgroundColor: T.accent }]} />
            )}
          </React.Fragment>
        ))}
      </View>

      <ScrollView contentContainerStyle={{ padding: 20 }} showsVerticalScrollIndicator={false}>
        {step === 0 && (
          <View>
            <Text style={styles.enrollHint}>
              Fill in the employee's basic details to begin enrollment.
            </Text>
            <Text style={styles.inputLabel}>FULL NAME</Text>
            <InputField icon="👤" placeholder="Enter full name" />
            <Text style={[styles.inputLabel, { marginTop: 14 }]}>EMPLOYEE ID</Text>
            <InputField icon="🪪" placeholder="e.g. EMP007" />
            <Text style={[styles.inputLabel, { marginTop: 14 }]}>DEPARTMENT</Text>
            <InputField icon="🏢" placeholder="Select department" />
            <Text style={[styles.inputLabel, { marginTop: 14 }]}>DESIGNATION</Text>
            <InputField icon="💼" placeholder="Enter designation" />
            <PrimaryBtn label="Next →" onPress={goNext} style={{ marginTop: 24 }} />
          </View>
        )}

        {step === 1 && (
          <View>
            <Text style={styles.enrollHint}>
              Capture a clear front-facing photo for biometric enrollment.
            </Text>
            <TouchableOpacity style={styles.captureCircle} onPress={goNext} activeOpacity={0.8}>
              <Text style={{ fontSize: 44 }}>📸</Text>
              <Text style={styles.captureText}>Tap to capture face</Text>
            </TouchableOpacity>
            <View style={styles.qualityChecks}>
              {['💡 Good Lighting', '🚫 No Accessories', '👁️ Look Straight'].map(q => (
                <View key={q} style={styles.qCheck}>
                  <Text style={{ fontSize: 10, color: T.text3, textAlign: 'center' }}>{q}</Text>
                </View>
              ))}
            </View>
            <PrimaryBtn label="Capture & Continue →" onPress={goNext} style={{ marginTop: 8 }} />
            <SecondaryBtn label="← Back" onPress={goBack} style={{ marginTop: 10 }} />
          </View>
        )}

        {step === 2 && (
          <View>
            <Text style={styles.enrollHint}>
              Review details before saving to the encrypted local database.
            </Text>
            <View style={styles.resultCard}>
              <View style={{ flexDirection: 'row', alignItems: 'center', gap: 14, marginBottom: 12 }}>
                <View style={styles.resultAvatar}><Text style={{ fontSize: 28 }}>👨🏽‍💼</Text></View>
                <View>
                  <Text style={styles.resultPersonName}>Aarav Sharma</Text>
                  <Text style={styles.resultPersonId}>EMP007 · Engineering</Text>
                </View>
              </View>
              {[
                { k: 'Designation',    v: 'Software Developer' },
                { k: 'Embedding Size', v: '128-dim · Ready', vc: T.accent },
                { k: 'Encrypted',      v: '✓ AES-256',       vc: T.accent },
              ].map((m, i) => (
                <View key={i} style={styles.metaRow}>
                  <Text style={styles.statusKey}>{m.k}</Text>
                  <Text style={[styles.metaVal, m.vc ? { color: m.vc } : {}]}>{m.v}</Text>
                </View>
              ))}
            </View>
            <PrimaryBtn label="Confirm & Save ✓" onPress={goNext} style={{ marginTop: 16 }} />
            <SecondaryBtn label="← Retake Photo" onPress={goBack} style={{ marginTop: 10 }} />
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
};


// 8. LOGS SCREEN

const logsData = [
  { name: 'Aarav Sharma · EMP001',  time: 'Today · 09:31 AM · Offline',      score: '98%',   type: 'success' },
  { name: 'Diya Patel · EMP002',    time: 'Today · 09:25 AM · Offline',      score: '96%',   type: 'success' },
  { name: 'Rohan Verma · EMP003',   time: 'Today · 09:19 AM · Offline',      score: 'Failed',type: 'fail'    },
  { name: 'Sneha Iyer · EMP004',    time: 'Today · 09:11 AM · Offline',      score: '97%',   type: 'success' },
  { name: 'Karan Singh · EMP005',   time: 'Yesterday · 05:44 PM · Pending',  score: 'Sync',  type: 'sync'    },
  { name: 'Meera Nair · EMP006',    time: 'Yesterday · 04:30 PM · Synced',   score: '95%',   type: 'success' },
];

const LogsScreen = () => {
  const [tab, setTab] = useState('All');
  const tabs = ['All', 'Verified', 'Failed', 'Sync'];

  const dotColor: Record<string, string> = {
    success: T.accent,
    fail:    T.danger,
    sync:    T.warn,
  };
  const scoreColor: Record<string, string> = {
    success: T.accent,
    fail:    T.danger,
    sync:    T.warn,
  };

  const visible = tab === 'All'
    ? logsData
    : logsData.filter(l => {
        if (tab === 'Verified') return l.type === 'success';
        if (tab === 'Failed')   return l.type === 'fail';
        if (tab === 'Sync')     return l.type === 'sync';
        return true;
      });

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: T.bg }}>
      <StatusBar barStyle="light-content" backgroundColor={T.bg} />
      <View style={styles.topHeader}>
        <Text style={styles.screenTitle}>Verification Logs</Text>
        <View style={styles.iconBtn}><Text style={{ fontSize: 16 }}>📤</Text></View>
      </View>

      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={{ padding: 20 }}>
        {/* Filter tabs */}
        <View style={{ flexDirection: 'row', gap: 8, marginBottom: 16 }}>
          {tabs.map(t => (
            <TouchableOpacity
              key={t}
              style={[styles.logTab, tab === t && styles.logTabActive]}
              onPress={() => setTab(t)}>
              <Text style={[styles.logTabText, tab === t && { color: T.accent }]}>{t}</Text>
            </TouchableOpacity>
          ))}
        </View>

        {visible.map((l, i) => (
          <View key={i} style={styles.logItem}>
            <View style={[styles.logDot, { backgroundColor: dotColor[l.type] }]} />
            <View style={{ flex: 1 }}>
              <Text style={styles.empName}>{l.name}</Text>
              <Text style={styles.empId}>{l.time}</Text>
            </View>
            <Text style={[styles.logScore, { color: scoreColor[l.type] }]}>{l.score}</Text>
          </View>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
};


// 9. SYNC SCREEN

const syncRecords = [
  { icon: '✅', name: 'Aarav Sharma · Verified',  time: '29 May 2026 · 09:31 AM' },
  { icon: '✅', name: 'Diya Patel · Verified',    time: '29 May 2026 · 09:25 AM' },
  { icon: '❌', name: 'Rohan Verma · Failed',     time: '29 May 2026 · 09:19 AM' },
  { icon: '✅', name: 'Sneha Iyer · Verified',    time: '29 May 2026 · 09:11 AM' },
];

const SyncScreen = ({ navigation }: any) => {
  const floatAnim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(floatAnim, { toValue: -8, duration: 1500, useNativeDriver: true }),
        Animated.timing(floatAnim, { toValue: 0,  duration: 1500, useNativeDriver: true }),
      ]),
    ).start();
  }, []);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: T.bg }}>
      <StatusBar barStyle="light-content" backgroundColor={T.bg} />
      <View style={styles.topHeader}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Text style={{ color: T.text, fontSize: 18 }}>←</Text>
        </TouchableOpacity>
        <Text style={styles.screenTitle}>Pending Sync</Text>
        <View style={{ width: 36 }} />
      </View>

      <ScrollView contentContainerStyle={{ padding: 20 }} showsVerticalScrollIndicator={false}>
        {/* Hero */}
        <View style={styles.syncHero}>
          <Animated.Text style={{ fontSize: 52, transform: [{ translateY: floatAnim }] }}>☁️</Animated.Text>
          <Text style={[styles.statNum, { color: T.warn, fontSize: 42 }]}>12</Text>
          <Text style={styles.syncHeroLabel}>
            Records pending upload to AWS S3.{'\n'}Will sync automatically when online.
          </Text>
        </View>

        {/* Offline warning */}
        <View style={[styles.offlineBanner, { borderRadius: 14, borderWidth: 1, borderColor: 'rgba(255,82,82,0.3)', marginBottom: 16 }]}>
          <View style={styles.offlineDot} />
          <Text style={[styles.offlineText, { flex: 1, flexWrap: 'wrap' }]}>
            You are offline. Records will sync when internet connection is restored.
          </Text>
        </View>

        <TouchableOpacity style={styles.syncBtn} activeOpacity={0.8}>
          <Text style={{ fontSize: 16 }}>🔄</Text>
          <Text style={styles.syncBtnText}>Sync Now (Requires Connection)</Text>
        </TouchableOpacity>

        <Text style={[styles.sectionTitle, { marginTop: 20, marginBottom: 12 }]}>PENDING RECORDS</Text>
        {syncRecords.map((r, i) => (
          <View key={i} style={styles.syncRecord}>
            <Text style={{ fontSize: 22 }}>{r.icon}</Text>
            <View style={{ flex: 1 }}>
              <Text style={styles.empName}>{r.name}</Text>
              <Text style={styles.empId}>{r.time}</Text>
            </View>
            <View style={styles.pendingTag}>
              <Text style={{ color: T.warn, fontSize: 10, fontWeight: '700' }}>Pending</Text>
            </View>
          </View>
        ))}
      </ScrollView>
    </SafeAreaView>
  );
};


// 10. REPORTS SCREEN

const barData = [
  { day: 'Mon', h: 40 },
  { day: 'Tue', h: 55 },
  { day: 'Wed', h: 48 },
  { day: 'Thu', h: 70 },
  { day: 'Fri', h: 62 },
  { day: 'Sat', h: 30 },
  { day: 'Sun', h: 20 },
];

const ReportsScreen = ({ navigation }: any) => {
  const barAnims = barData.map(() => useRef(new Animated.Value(0)).current);

  useEffect(() => {
    Animated.stagger(
      80,
      barAnims.map((a, i) =>
        Animated.timing(a, { toValue: barData[i].h, duration: 600, useNativeDriver: false }),
      ),
    ).start();
  }, []);

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: T.bg }}>
      <StatusBar barStyle="light-content" backgroundColor={T.bg} />
      <View style={styles.topHeader}>
        <TouchableOpacity style={styles.backBtn} onPress={() => navigation.goBack()}>
          <Text style={{ color: T.text, fontSize: 18 }}>←</Text>
        </TouchableOpacity>
        <Text style={styles.screenTitle}>Reports</Text>
        <View style={styles.iconBtn}><Text style={{ fontSize: 16 }}>📤</Text></View>
      </View>

      <ScrollView contentContainerStyle={{ padding: 20 }} showsVerticalScrollIndicator={false}>
        
        <View style={{ flexDirection: 'row', gap: 10, marginBottom: 16 }}>
          {[
            { val: '156', label: 'Total', sub: 'This week', vc: T.text  },
            { val: '142', label: 'Success', sub: '91%',     vc: T.accent },
            { val: '14',  label: 'Failed',  sub: '9%',      vc: T.danger },
          ].map(s => (
            <View key={s.label} style={[styles.resultCard, { flex: 1, padding: 14 }]}>
              <Text style={[styles.statNum, { fontSize: 22, color: s.vc }]}>{s.val}</Text>
              <Text style={styles.empId}>{s.label}</Text>
              <Text style={{ fontSize: 11, color: T.accent, fontWeight: '600' }}>{s.sub}</Text>
            </View>
          ))}
        </View>

        
        <View style={styles.chartCard}>
          <Text style={styles.chartTitle}>Verification Trend — This Week</Text>
          <View style={styles.barChart}>
            {barData.map((b, i) => (
              <View key={b.day} style={styles.barWrap}>
                <Animated.View
                  style={[
                    styles.bar,
                    { height: barAnims[i] },
                  ]}
                />
                <Text style={styles.barDay}>{b.day}</Text>
              </View>
            ))}
          </View>
        </View>

        
        <View style={styles.statusCard}>
          <Text style={[styles.sectionTitle, { marginBottom: 0, paddingBottom: 12, borderBottomWidth: 1, borderBottomColor: T.border }]}>
            PERFORMANCE METRICS
          </Text>
          {[
            { k: 'Avg. Inference Time',       v: '187ms',  vc: T.accent },
            { k: 'Avg. Confidence Score',      v: '96.4%',  vc: T.accent },
            { k: 'Liveness Pass Rate',         v: '98.7%',  vc: T.accent },
            { k: 'Spoofing Attempts Blocked',  v: '3',      vc: T.warn   },
          ].map((m, i) => (
            <View key={i} style={[styles.statusRow, i === 3 && { borderBottomWidth: 0 }]}>
              <Text style={styles.statusKey}>{m.k}</Text>
              <Text style={[styles.statusVal, { color: m.vc }]}>{m.v}</Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
};


// 11. PROFILE SCREEN

const ProfileScreen = ({ navigation }: any) => {
  const { role } = useContext(AuthContext);

  const menuItems = [
    { icon: '🏢', label: 'Organisation Settings' },
    { icon: '🔑', label: 'Change Password'       },
    { icon: '⚙️', label: 'App Settings'          },
    { icon: '🔒', label: 'Security & Encryption' },
    { icon: 'ℹ️', label: 'About EdgeAuth'        },
  ];

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: T.bg }}>
      <StatusBar barStyle="light-content" backgroundColor={T.bg} />

      
      <View style={styles.profileHero}>
        <View style={styles.profileAvatarLarge}>
          <Text style={{ fontSize: 40 }}>👨🏽‍💻</Text>
        </View>
        <Text style={styles.profileName}>Admin User</Text>
        <Text style={styles.profileEmail}>admin@bennettuniversity.edu.in</Text>
        <View style={styles.roleBadge}>
          <Text style={{ fontSize: 11, color: T.accent, fontWeight: '700' }}>{role}</Text>
        </View>
      </View>

      <ScrollView contentContainerStyle={{ padding: 20 }} showsVerticalScrollIndicator={false}>
        {menuItems.map(item => (
          <TouchableOpacity key={item.label} style={styles.profileMenuItem} activeOpacity={0.8}>
            <View style={styles.pmiIcon}>
              <Text style={{ fontSize: 18 }}>{item.icon}</Text>
            </View>
            <Text style={styles.pmiLabel}>{item.label}</Text>
            <Text style={{ color: T.text3, fontSize: 18 }}>›</Text>
          </TouchableOpacity>
        ))}

        
        <TouchableOpacity
          style={[styles.profileMenuItem, styles.logoutItem]}
          onPress={() => navigation.replace('Auth')}
          activeOpacity={0.8}>
          <View style={[styles.pmiIcon, { backgroundColor: 'rgba(255,82,82,0.1)' }]}>
            <Text style={{ fontSize: 18 }}>🚪</Text>
          </View>
          <Text style={[styles.pmiLabel, { color: T.danger }]}>Logout</Text>
          <Text style={{ color: T.danger, fontSize: 18 }}>›</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
};


// BOTTOM TAB BAR  

const TabBar = ({ state, navigation }: any) => {
  const tabs = [
    { name: 'Dashboard', icon: '🏠', label: 'Home'      },
    { name: 'Employees', icon: '👥', label: 'Employees' },
    { name: 'Logs',      icon: '📋', label: 'Logs'      },
    { name: 'Profile',   icon: '👤', label: 'Profile'   },
  ];

  return (
    <View style={styles.tabBar}>
      {tabs.slice(0, 2).map((t, i) => {
        const focused = state.index === i;
        return (
          <TouchableOpacity
            key={t.name}
            style={[styles.tabItem, focused && styles.tabItemActive]}
            onPress={() => navigation.navigate(t.name)}
            activeOpacity={0.8}>
            <Text style={{ fontSize: 20 }}>{t.icon}</Text>
            <Text style={[styles.tabLabel, focused && { color: T.accent }]}>{t.label}</Text>
          </TouchableOpacity>
        );
      })}

      
      <TouchableOpacity
        style={styles.tabFab}
        onPress={() => navigation.navigate('Verify')}
        activeOpacity={0.85}>
        <Text style={{ fontSize: 26 }}>🛡️</Text>
      </TouchableOpacity>

      {tabs.slice(2).map((t, i) => {
        const focused = state.index === i + 2;
        return (
          <TouchableOpacity
            key={t.name}
            style={[styles.tabItem, focused && styles.tabItemActive]}
            onPress={() => navigation.navigate(t.name)}
            activeOpacity={0.8}>
            <Text style={{ fontSize: 20 }}>{t.icon}</Text>
            <Text style={[styles.tabLabel, focused && { color: T.accent }]}>{t.label}</Text>
          </TouchableOpacity>
        );
      })}
    </View>
  );
};


// MAIN TABS NAVIGATOR

const MainTabs = () => (
  <Tab.Navigator
    tabBar={props => <TabBar {...props} />}
    screenOptions={{ headerShown: false }}>
    <Tab.Screen name="Dashboard" component={DashboardScreen} />
    <Tab.Screen name="Employees" component={EmployeesScreen} />
    <Tab.Screen name="Logs"      component={LogsScreen}      />
    <Tab.Screen name="Profile"   component={ProfileScreen}   />
  </Tab.Navigator>
);


// MAIN APP STACK  

const MainAppNavigator = () => (
  <MainStack.Navigator screenOptions={{ headerShown: false }}>
    <MainStack.Screen name="Tabs"    component={MainTabs}    />
    <MainStack.Screen name="Verify"  component={VerifyScreen}  options={{ presentation: 'modal' }} />
    <MainStack.Screen name="Result"  component={ResultScreen}  />
    <MainStack.Screen name="Enroll"  component={EnrollScreen}  />
    <MainStack.Screen name="Sync"    component={SyncScreen}    />
    <MainStack.Screen name="Reports" component={ReportsScreen} />
  </MainStack.Navigator>
);


// AUTH STACK

const AuthNavigator = () => (
  <AuthStack.Navigator screenOptions={{ headerShown: false }}>
    <AuthStack.Screen name="Splash" component={SplashScreen} />
    <AuthStack.Screen name="Login"  component={LoginScreen}  />
  </AuthStack.Navigator>
);


// ROOT

const RootNavigator = () => (
  <RootStack.Navigator screenOptions={{ headerShown: false }}>
    <RootStack.Screen name="Auth"    component={AuthNavigator}    />
    <RootStack.Screen name="MainApp" component={MainAppNavigator} />
  </RootStack.Navigator>
);


// APP ENTRY

export default function App() {
  const [role, setRole] = useState('Admin');

  return (
    <AuthContext.Provider value={{ role, setRole }}>
      <NavigationContainer>
        <RootNavigator />
      </NavigationContainer>
    </AuthContext.Provider>
  );
}


// STYLESHEET

const styles = StyleSheet.create({

  
  splashContainer: {
    flex: 1,
    backgroundColor: T.bg,
    alignItems: 'center',
    justifyContent: 'center',
  },
  gridLine: {
    position: 'absolute',
    backgroundColor: 'rgba(0,230,118,0.04)',
  },
  ring: {
    borderRadius: 999,
    borderWidth: 1,
    borderColor: 'rgba(0,230,118,0.25)',
  },
  shieldWrap: {
    backgroundColor: '#161e16',
    borderWidth: 1.5,
    borderColor: T.accent3,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 20,
    shadowColor: T.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.3,
    shadowRadius: 20,
    elevation: 10,
  },
  splashAppName: {
    fontSize: 38,
    fontWeight: '900',
    color: T.text,
    letterSpacing: -1,
  },
  splashTagline: {
    fontSize: 11,
    color: T.text3,
    letterSpacing: 3,
    marginTop: 6,
    marginBottom: 50,
  },
  splashSub: {
    fontSize: 13,
    color: T.text2,
    marginTop: -40,
  },
  initRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    marginTop: 60,
  },
  spinnerCircle: {
    width: 16,
    height: 16,
    borderRadius: 8,
    borderWidth: 2,
    borderColor: T.border,
    borderTopColor: T.accent,
  },
  initText: {
    fontSize: 12,
    color: T.text3,
  },
  versionText: {
    position: 'absolute',
    bottom: 40,
    fontSize: 11,
    color: T.text3,
  },

  
  loginScroll: {
    padding: 24,
    paddingTop: 40,
  },
  loginTitle: {
    fontSize: 28,
    fontWeight: '900',
    color: T.text,
    marginBottom: 4,
  },
  loginSub: {
    fontSize: 13,
    color: T.text3,
    marginBottom: 28,
  },
  roleRow: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 28,
  },
  roleTab: {
    flex: 1,
    alignItems: 'center',
    gap: 6,
    paddingVertical: 12,
    paddingHorizontal: 6,
    borderRadius: 14,
    borderWidth: 1.5,
    borderColor: T.border,
    backgroundColor: T.card,
  },
  roleTabActive: {
    borderColor: T.accent,
    backgroundColor: 'rgba(0,230,118,0.08)',
  },
  roleTabLabel: {
    fontSize: 11,
    color: T.text2,
    fontWeight: '700',
    textAlign: 'center',
  },
  inputLabel: {
    fontSize: 11,
    color: T.text3,
    letterSpacing: 0.5,
    fontWeight: '700',
    marginBottom: 8,
  },
  inputWrap: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: T.card,
    borderWidth: 1.5,
    borderColor: T.border,
    borderRadius: 14,
    paddingHorizontal: 14,
    height: 52,
  },
  inputIcon: {
    fontSize: 16,
    marginRight: 10,
  },
  input: {
    flex: 1,
    color: T.text,
    fontSize: 14,
    height: '100%',
  },
  dividerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginVertical: 20,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: T.border,
  },
  dividerText: {
    fontSize: 12,
    color: T.text3,
  },
  qrBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    backgroundColor: T.card2,
    borderWidth: 1.5,
    borderColor: T.border,
    borderRadius: 16,
    paddingVertical: 14,
  },
  qrBtnText: {
    color: T.text,
    fontSize: 14,
    fontWeight: '600',
  },

  
  primaryBtn: {
    backgroundColor: T.accent,
    borderRadius: 16,
    paddingVertical: 16,
    alignItems: 'center',
    shadowColor: T.accent,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.35,
    shadowRadius: 12,
    elevation: 6,
  },
  primaryBtnText: {
    color: '#0a0f0a',
    fontSize: 15,
    fontWeight: '800',
    letterSpacing: 0.4,
  },
  secondaryBtn: {
    backgroundColor: T.card,
    borderRadius: 16,
    paddingVertical: 14,
    alignItems: 'center',
    borderWidth: 1.5,
    borderColor: T.border,
  },
  secondaryBtnText: {
    color: T.text,
    fontSize: 14,
    fontWeight: '600',
  },

  
  offlineBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 20,
    paddingVertical: 8,
    backgroundColor: 'rgba(255,82,82,0.1)',
    borderBottomWidth: 1,
    borderBottomColor: 'rgba(255,82,82,0.3)',
  },
  offlineDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: T.danger,
  },
  offlineText: {
    fontSize: 11,
    color: T.danger,
    fontWeight: '700',
    letterSpacing: 0.4,
    flex: 1,
  },
  offlineCount: {
    fontSize: 11,
    color: T.warn,
    fontWeight: '700',
  },

  
  topHeader: {
    height: 60,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
  },
  screenTitle: {
    fontSize: 18,
    fontWeight: '800',
    color: T.text,
  },
  backBtn: {
    width: 36,
    height: 36,
    backgroundColor: T.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: T.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  iconBtn: {
    width: 38,
    height: 38,
    backgroundColor: T.card,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: T.border,
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  notifBadge: {
    position: 'absolute',
    top: -4,
    right: -4,
    width: 16,
    height: 16,
    backgroundColor: T.danger,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },

  
  dashHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 8,
  },
  dashGreeting: { fontSize: 13, color: T.text3 },
  dashName:     { fontSize: 24, fontWeight: '900', color: T.text },
  dashOrg:      { fontSize: 12, color: T.accent, fontWeight: '700', marginBottom: 20 },

  statCard: {
    marginHorizontal: 20,
    marginBottom: 20,
    backgroundColor: '#1a3a1a',
    borderWidth: 1,
    borderColor: '#2a4a2a',
    borderRadius: 20,
    padding: 20,
    overflow: 'hidden',
  },
  statLabel: { fontSize: 12, color: T.text3, marginBottom: 4 },
  statNum:   { fontSize: 42, fontWeight: '900', color: T.text, lineHeight: 48 },
  statSub:   { fontSize: 11, color: T.accent, marginTop: 4 },
  sparklineWrap: {
    position: 'absolute',
    right: 20,
    bottom: 20,
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 3,
    height: 40,
  },
  sparkBar: {
    width: 8,
    backgroundColor: T.accent,
    borderRadius: 2,
  },

  sectionTitle: {
    fontSize: 12,
    fontWeight: '800',
    color: T.text2,
    letterSpacing: 0.5,
    marginBottom: 14,
    marginTop: 4,
  },
  quickGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 20,
    gap: 10,
    marginBottom: 20,
  },
  quickItem: {
    width: (SW - 60) / 3,
    backgroundColor: T.card,
    borderWidth: 1,
    borderColor: T.border,
    borderRadius: 16,
    paddingVertical: 14,
    alignItems: 'center',
    gap: 8,
    position: 'relative',
  },
  quickLabel: {
    fontSize: 10,
    color: T.text2,
    fontWeight: '700',
    textAlign: 'center',
  },
  quickBadge: {
    position: 'absolute',
    top: 8,
    right: 8,
    backgroundColor: T.danger,
    borderRadius: 8,
    paddingHorizontal: 5,
    paddingVertical: 2,
  },
  statusCard: {
    marginHorizontal: 20,
    marginBottom: 20,
    backgroundColor: T.card,
    borderWidth: 1,
    borderColor: T.border,
    borderRadius: 20,
    paddingHorizontal: 20,
  },
  statusRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: T.border,
  },
  statusKey: { fontSize: 13, color: T.text2 },
  statusVal: { fontSize: 13, fontWeight: '700' },


  cameraBg: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: '#0d1f0d',
  },
  scanStripe: {
    position: 'absolute',
    left: 0,
    right: 0,
    height: 2,
    backgroundColor: 'rgba(0,230,118,0.015)',
  },
  verifyTopBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 8,
    paddingBottom: 8,
  },
  verifyBackBtn: {
    width: 36,
    height: 36,
    backgroundColor: 'rgba(255,255,255,0.12)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  verifyTitle: {
    fontSize: 17,
    fontWeight: '800',
    color: T.white,
  },
  flashBtn: {
    width: 36,
    height: 36,
    backgroundColor: 'rgba(255,255,255,0.12)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center',
    justifyContent: 'center',
  },
  faceGuideText: {
    textAlign: 'center',
    fontSize: 13,
    color: T.accent,
    letterSpacing: 0.4,
    marginBottom: 8,
  },
  ovalOuter: {
    width: 240,
    height: 290,
    borderRadius: 120,
    borderWidth: 2,
    borderColor: T.accent,
    shadowColor: T.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.6,
    shadowRadius: 16,
    elevation: 10,
    overflow: 'hidden',
  },
  ovalInner: {
    flex: 1,
    position: 'relative',
    overflow: 'hidden',
  },
  facePlaceholder: {
    ...StyleSheet.absoluteFillObject,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(30,30,30,0.8)',
  },
  scanLineAnim: {
    position: 'absolute',
    left: 10,
    right: 10,
    height: 2,
    backgroundColor: T.accent,
    borderRadius: 1,
    shadowColor: T.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.9,
    shadowRadius: 6,
    elevation: 4,
  },
  corner: {
    position: 'absolute',
    width: 20,
    height: 20,
    borderColor: T.accent,
    borderStyle: 'solid',
  },
  cornerTL: { top: -1,  left: -1,  borderTopWidth: 3, borderLeftWidth: 3,   borderTopLeftRadius: 4 },
  cornerTR: { top: -1,  right: -1, borderTopWidth: 3, borderRightWidth: 3,  borderTopRightRadius: 4 },
  cornerBL: { bottom: -1, left: -1,  borderBottomWidth: 3, borderLeftWidth: 3,  borderBottomLeftRadius: 4 },
  cornerBR: { bottom: -1, right: -1, borderBottomWidth: 3, borderRightWidth: 3, borderBottomRightRadius: 4 },

  livenessPanel: {
    backgroundColor: 'rgba(10,15,10,0.94)',
    borderTopWidth: 1,
    borderTopColor: 'rgba(0,230,118,0.25)',
    paddingHorizontal: 24,
    paddingVertical: 20,
    paddingBottom: Platform.OS === 'ios' ? 34 : 20,
  },
  challengeLabel: {
    fontSize: 11,
    color: T.accent,
    letterSpacing: 1.5,
    textTransform: 'uppercase',
    marginBottom: 4,
  },
  challengeText: {
    fontSize: 22,
    fontWeight: '900',
    color: T.white,
    marginBottom: 12,
  },
  chDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    backgroundColor: T.border,
  },
  chDotDone: {
    backgroundColor: T.accent,
  },
  chDotActive: {
    backgroundColor: T.accent,
    shadowColor: T.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 6,
    elevation: 4,
  },
  challengeHint: {
    fontSize: 12,
    color: T.text2,
    marginBottom: 4,
  },

  
  resultTopBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingTop: 8,
    height: 60,
  },
  resultIconWrap: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: 'rgba(0,230,118,0.1)',
    borderWidth: 2,
    borderColor: T.accent,
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'center',
    marginBottom: 16,
    shadowColor: T.accent,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.4,
    shadowRadius: 20,
    elevation: 8,
  },
  resultTitle: {
    fontSize: 24,
    fontWeight: '900',
    color: T.accent,
    textAlign: 'center',
    marginBottom: 4,
  },
  resultSub: {
    fontSize: 13,
    color: T.text2,
    textAlign: 'center',
    marginBottom: 24,
  },
  resultCard: {
    backgroundColor: T.card,
    borderWidth: 1,
    borderColor: T.border,
    borderRadius: 20,
    padding: 20,
    marginBottom: 8,
  },
  resultAvatar: {
    width: 52,
    height: 52,
    backgroundColor: T.card2,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: T.border,
  },
  resultPersonName: {
    fontSize: 17,
    fontWeight: '800',
    color: T.text,
  },
  resultPersonId: {
    fontSize: 12,
    color: T.text3,
  },
  confidenceTrack: {
    height: 6,
    backgroundColor: T.border,
    borderRadius: 3,
    overflow: 'hidden',
  },
  confidenceFill: {
    height: '100%',
    backgroundColor: T.accent,
    borderRadius: 3,
  },
  metaRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingVertical: 8,
    borderTopWidth: 1,
    borderTopColor: T.border,
  },
  metaVal: {
    fontSize: 12,
    color: T.text,
    fontWeight: '700',
  },

  
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    backgroundColor: T.card,
    borderWidth: 1,
    borderColor: T.border,
    borderRadius: 14,
    paddingHorizontal: 14,
    height: 48,
    marginBottom: 12,
  },
  searchInput: {
    flex: 1,
    color: T.text,
    fontSize: 14,
  },
  addEmpBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    borderWidth: 1.5,
    borderColor: 'rgba(0,230,118,0.35)',
    borderStyle: 'dashed',
    borderRadius: 16,
    paddingVertical: 14,
    backgroundColor: 'rgba(0,230,118,0.04)',
    marginBottom: 14,
  },
  addEmpText: {
    color: T.accent,
    fontSize: 14,
    fontWeight: '700',
  },
  empItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    backgroundColor: T.card,
    borderWidth: 1,
    borderColor: T.border,
    borderRadius: 16,
    padding: 14,
    marginBottom: 10,
  },
  empAvatar: {
    width: 44,
    height: 44,
    backgroundColor: '#1e3a1e',
    borderRadius: 14,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: T.accent3,
  },
  empName: { fontSize: 14, fontWeight: '700', color: T.text },
  empId:   { fontSize: 11, color: T.text3 },
  empStatusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },

  
  stepsRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 0,
    paddingVertical: 16,
    paddingHorizontal: 40,
  },
  stepBubble: {
    width: 28,
    height: 28,
    borderRadius: 14,
    borderWidth: 2,
    borderColor: T.border,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stepBubbleActive: {
    borderColor: T.accent,
    backgroundColor: 'rgba(0,230,118,0.08)',
  },
  stepBubbleDone: {
    backgroundColor: T.accent,
    borderColor: T.accent,
  },
  stepLine: {
    flex: 1,
    height: 2,
    backgroundColor: T.border,
    marginHorizontal: 4,
  },
  enrollHint: {
    fontSize: 13,
    color: T.text3,
    lineHeight: 20,
    marginBottom: 20,
  },
  captureCircle: {
    width: 200,
    height: 200,
    borderRadius: 100,
    backgroundColor: T.card,
    borderWidth: 2,
    borderColor: T.accent3,
    borderStyle: 'dashed',
    alignItems: 'center',
    justifyContent: 'center',
    alignSelf: 'center',
    gap: 10,
    marginBottom: 24,
  },
  captureText: {
    fontSize: 12,
    color: T.text3,
    textAlign: 'center',
  },
  qualityChecks: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: 24,
    marginBottom: 24,
  },
  qCheck: { alignItems: 'center', gap: 4, width: 80 },

  
  logTab: {
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: T.border,
    backgroundColor: T.card,
  },
  logTabActive: {
    backgroundColor: 'rgba(0,230,118,0.1)',
    borderColor: T.accent,
  },
  logTabText: { fontSize: 12, fontWeight: '700', color: T.text3 },
  logItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: T.card,
    borderWidth: 1,
    borderColor: T.border,
    borderRadius: 14,
    padding: 14,
    marginBottom: 8,
  },
  logDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  logScore: { fontSize: 13, fontWeight: '800' },

  
  syncHero: {
    backgroundColor: T.card,
    borderWidth: 1,
    borderColor: T.border,
    borderRadius: 20,
    padding: 28,
    alignItems: 'center',
    gap: 10,
    marginBottom: 16,
  },
  syncHeroLabel: {
    fontSize: 13,
    color: T.text3,
    textAlign: 'center',
    lineHeight: 20,
  },
  syncBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 10,
    backgroundColor: T.card2,
    borderWidth: 1.5,
    borderColor: T.warn,
    borderRadius: 16,
    paddingVertical: 16,
  },
  syncBtnText: {
    color: T.warn,
    fontSize: 14,
    fontWeight: '700',
  },
  syncRecord: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    backgroundColor: T.card,
    borderWidth: 1,
    borderColor: T.border,
    borderRadius: 14,
    padding: 14,
    marginBottom: 8,
  },
  pendingTag: {
    backgroundColor: 'rgba(255,171,64,0.1)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },

  
  chartCard: {
    backgroundColor: T.card,
    borderWidth: 1,
    borderColor: T.border,
    borderRadius: 20,
    padding: 16,
    marginBottom: 16,
  },
  chartTitle: {
    fontSize: 13,
    color: T.text2,
    fontWeight: '700',
    marginBottom: 14,
  },
  barChart: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    height: 80,
    gap: 8,
  },
  barWrap: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'flex-end',
    height: '100%',
    gap: 4,
  },
  bar: {
    width: '100%',
    backgroundColor: T.accent,
    borderRadius: 4,
    opacity: 0.85,
  },
  barDay: { fontSize: 9, color: T.text3 },

  
  profileHero: {
    alignItems: 'center',
    paddingTop: 20,
    paddingBottom: 16,
    paddingHorizontal: 20,
  },
  profileAvatarLarge: {
    width: 80,
    height: 80,
    backgroundColor: '#1e3a1e',
    borderRadius: 24,
    borderWidth: 2,
    borderColor: T.accent3,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  profileName:  { fontSize: 20, fontWeight: '900', color: T.text },
  profileEmail: { fontSize: 13, color: T.text3, marginBottom: 6 },
  roleBadge: {
    backgroundColor: 'rgba(0,230,118,0.1)',
    paddingHorizontal: 16,
    paddingVertical: 5,
    borderRadius: 20,
  },
  profileMenuItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    backgroundColor: T.card,
    borderWidth: 1,
    borderColor: T.border,
    borderRadius: 16,
    padding: 16,
    marginBottom: 10,
  },
  logoutItem: {
    backgroundColor: 'rgba(255,82,82,0.05)',
    borderColor: 'rgba(255,82,82,0.2)',
  },
  pmiIcon: {
    width: 36,
    height: 36,
    backgroundColor: T.card2,
    borderRadius: 10,
    alignItems: 'center',
    justifyContent: 'center',
  },
  pmiLabel: { flex: 1, fontSize: 14, color: T.text, fontWeight: '500' },

   
  tabBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: T.bg2,
    borderTopWidth: 1,
    borderTopColor: T.border,
    paddingBottom: Platform.OS === 'ios' ? 20 : 8,
    paddingTop: 8,
    paddingHorizontal: 8,
    height: Platform.OS === 'ios' ? 82 : 64,
  },
  tabItem: {
    flex: 1,
    alignItems: 'center',
    gap: 3,
    paddingVertical: 6,
    borderRadius: 14,
  },
  tabItemActive: {
    backgroundColor: 'rgba(0,230,118,0.08)',
  },
  tabLabel: {
    fontSize: 10,
    color: T.text3,
    fontWeight: '600',
  },
  tabFab: {
    width: 56,
    height: 56,
    borderRadius: 18,
    backgroundColor: T.accent,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: T.accent,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.5,
    shadowRadius: 12,
    elevation: 8,
    marginHorizontal: 8,
  },
});
