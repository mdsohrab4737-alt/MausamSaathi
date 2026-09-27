import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { supabase } from './supabaseClient';
import './styles.css';

const API = 'http://127.0.0.1:8000';
// step21-map-block-filter-boundaries-installed
// step20-account-ui-polished-v8
// step19-auth-persistence-installed
// step20-farmer-first-location-installed
const HORIZONS = [7, 14, 21, 30];

const LANGUAGES = [
  { code: 'en', label: 'English' },
  { code: 'hi', label: 'हिंदी' },
];

const UI = {
  en: {
    language: 'Language',
    days: 'Days',
    confidence: 'Confidence',
    tagline: 'Hyperlocal Monsoon Intelligence',
    headerTag: 'Farmer-first',
    heroEyebrow: 'FROM CLIMATE SIGNALS TO FARM ACTION',
    heroTitle: 'Local monsoon risk, one selection away.',
    heroText: 'Select a location and forecast horizon to see probabilistic onset, break and heavy-rain risk.',
    workTitle: 'How does it work?',
    climate: 'Climate Signals', regional: 'Regional Weather', ai: 'AI Forecasting', local: 'Local Advisory',
    rainTempHum: 'Rainfall • Temperature • Humidity',
    largeLocal: 'Large-scale → Local',
    blockPanchayat: 'Block / Panchayat',
    selectLocation: '1. Select location', state: 'State', district: 'District', block: 'Block', panchayat: 'Panchayat',
    selectState: 'Select state', selectDistrict: 'Select district', selectBlock: 'Select block', selectPanchayat: 'Select panchayat',
    horizon: '2. Forecast horizon', viewForecast: 'View Forecast',
    readyTitle: 'Ready to view the forecast', readyText: 'Choose a location above, then open a 7–30 day forecast.',
    outlook: '3. Probabilistic outlook', demoView: 'Demo view',
    onset: 'MONSOON ONSET', breakPhase: 'BREAK PHASE', heavyRain: 'HEAVY RAIN', anomaly: 'RAINFALL ANOMALY',
    riskMap: '4. Colour-coded risk map', riskMapHead: 'Heavy Rain Risk', selectedArea: 'Selected area', low: 'Low', moderate: 'Moderate', high: 'High', veryHigh: 'Very High',
    mapIndicator: 'Map indicator', mapScope: 'Map scope', selectedPanchayat: 'Selected Panchayat', allPanchayats: 'All Panchayats',
    mapProbability: 'Probability view',
    rawProbability: 'Raw model output',
    calibratedProbability: 'Calibrated probability',
    snapshotDate: 'ML prediction date',
    mapLoading: 'Loading real Panchayat boundaries and ML probabilities…',
    mapError: 'Real spatial map could not be loaded.',
    mapHindcastNote: 'Current forecast snapshot • Prediction date shown above • 486 Gram Panchayat boundaries',
    selectedProbability: 'Selected probability',
    exampleMap: 'Map colour shows the selected risk. Click a Panchayat to see its exact percentage. Areas without a Gram Panchayat boundary remain as map background.',
    onsetSmall: 'Onset', breakSmall: 'Break', heavySmall: 'Heavy Rain', horizonSmall: 'Horizon',
    advisory: '5. Crop advisory', advisoryTitle: 'Turn probability into action', refreshAdvisory: 'Refresh Advisory',
    delivery: '6. Last-mile delivery', smsText: 'Text alert • regional language', whatsappText: 'Rich alert • ready to send', voiceText: 'Regional-language voice advisory', manyChannels: 'One forecast → many channels', manyChannelsText: 'Designed for farmers with different connectivity and literacy needs.', messagePreview: 'Message preview',
    sos: '7. Emergency / SOS', sosTitle: 'Escalate urgent local situations', sosText: 'A safety layer for urgent weather-related assistance through configured local response contacts.', sendSos: 'Send SOS Alert', supportCall: 'Call Support', sosNote: 'Demo flow: production can connect configured local assistance numbers and escalation services.',
    demoDisclaimer: 'Current ML spatial snapshot • Prediction date shown above • 486 mapped Gram Panchayat boundaries',
  },
  hi: {
    language: 'भाषा',
    days: 'दिन',
    confidence: 'विश्वसनीयता',
    tagline: 'स्थानीय मानसून जानकारी', headerTag: 'किसान के लिए',
    heroEyebrow: 'जलवायु संकेतों से किसान की कार्रवाई तक',
    heroTitle: 'स्थानीय मौसम जोखिम, सही सलाह एक चयन दूर।',
    heroText: 'अपना क्षेत्र और पूर्वानुमान अवधि चुनें और मानसून आगमन, विराम तथा भारी वर्षा का संभावित जोखिम देखें।',
    workTitle: 'हम कैसे काम करते हैं?', climate: 'जलवायु संकेत', regional: 'स्थानीय मौसम', ai: 'AI पूर्वानुमान', local: 'स्थानीय सलाह',
    rainTempHum: 'वर्षा • तापमान • आर्द्रता', largeLocal: 'बड़े पैमाने → स्थानीय', blockPanchayat: 'ब्लॉक / पंचायत',
    selectLocation: '1. अपना स्थान चुनें', state: 'राज्य', district: 'जिला', block: 'ब्लॉक', panchayat: 'पंचायत',
    selectState: 'राज्य चुनें', selectDistrict: 'जिला चुनें', selectBlock: 'ब्लॉक चुनें', selectPanchayat: 'पंचायत चुनें',
    horizon: '2. पूर्वानुमान अवधि', viewForecast: 'पूर्वानुमान देखें',
    readyTitle: 'पूर्वानुमान देखने के लिए तैयार', readyText: 'ऊपर अपना क्षेत्र चुनें और 7–30 दिनों का पूर्वानुमान देखें।',
    outlook: '3. संभावित पूर्वानुमान', demoView: 'डेमो दृश्य',
    onset: 'मानसून आगमन', breakPhase: 'मानसून विराम', heavyRain: 'भारी वर्षा', anomaly: 'वर्षा असामान्यता',
    riskMap: '4. रंग-कोडित जोखिम मानचित्र', riskMapHead: 'भारी वर्षा जोखिम', selectedArea: 'चयनित क्षेत्र', low: 'कम', moderate: 'मध्यम', high: 'उच्च', veryHigh: 'बहुत उच्च',
    mapIndicator: 'मानचित्र संकेतक', mapScope: 'मानचित्र क्षेत्र', selectedPanchayat: 'चयनित पंचायत', allPanchayats: 'सभी पंचायतें',
    mapProbability: 'संभाव्यता दृश्य',
    rawProbability: 'Raw',
    calibratedProbability: 'Calibrated',
    snapshotDate: 'ML पूर्वानुमान तिथि',
    mapLoading: 'वास्तविक पंचायत सीमाएँ और ML संभाव्यताएँ लोड हो रही हैं…',
    mapError: 'वास्तविक स्थानिक मानचित्र लोड नहीं हो सका।',
    mapHindcastNote: 'वास्तविक पंचायत सीमाएँ • ऐतिहासिक hindcast • लाइव नहीं',
    selectedProbability: 'चयनित संभाव्यता',
    exampleMap: 'ऐतिहासिक स्थानिक hindcast snapshot • ML probabilities FastAPI से • 486 स्थानीय निकाय सीमाएँ।',
    onsetSmall: 'आगमन', breakSmall: 'विराम', heavySmall: 'भारी वर्षा', horizonSmall: 'अवधि',
    advisory: '5. फसल सलाह', advisoryTitle: 'संभावना को कार्रवाई में बदलें', refreshAdvisory: 'सलाह ताज़ा करें',
    delivery: '6. किसान तक पहुँच', smsText: 'टेक्स्ट अलर्ट • क्षेत्रीय भाषा', whatsappText: 'रिच अलर्ट • भेजने के लिए तैयार', voiceText: 'क्षेत्रीय भाषा में वॉइस सलाह', manyChannels: 'एक पूर्वानुमान → कई माध्यम', manyChannelsText: 'अलग कनेक्टिविटी और साक्षरता जरूरतों वाले किसानों के लिए।', messagePreview: 'संदेश पूर्वावलोकन',
    sos: '7. आपातकालीन सहायता', sosTitle: 'जरूरत पड़ने पर त्वरित मदद', sosText: 'गंभीर मौसम स्थिति में स्थानीय सहायता के लिए SOS और सहायता-कॉल प्रवाह।', sendSos: 'SOS अलर्ट भेजें', supportCall: 'सहायता कॉल', sosNote: 'डेमो प्रवाह: प्रोडक्शन में स्थानीय सहायता नंबर और escalation services जोड़ी जा सकती हैं।',
    demoDisclaimer: 'ऐतिहासिक स्थानिक hindcast • लाइव operational forecast नहीं',
  },
};

function t(lang, key) {
  return UI[lang]?.[key] ?? UI.en[key] ?? key;
}


function onsetDisplayValue(selectedMlSummary, forecast, spatialMapPredictionDate) {
  const phaseDate =
    spatialMapPredictionDate ||
    forecast?.snapshot_date ||
    forecast?.forecast?.snapshot_date ||
    forecast?.forecast?.issue_date;
  if (!isOnsetWindowOpen(phaseDate)) return '—';
  return selectedMlSummary?.onset != null
    ? `${(selectedMlSummary.onset * 100).toFixed(1)}%`
    : `${forecast.forecast.onset_probability}%`;
}

function getMonsoonPhase(dateValue) {
  if (!dateValue) return "UNKNOWN";
  const d = new Date(`${String(dateValue).slice(0, 10)}T00:00:00`);
  if (Number.isNaN(d.getTime())) return "UNKNOWN";
  const month = d.getMonth() + 1;
  if (month < 6) return "PRE";
  if (month <= 8) return "MID";
  return "POST";
}

function isOnsetWindowOpen(dateValue) {
  if (!dateValue) return false;
  const d = new Date(`${String(dateValue).slice(0, 10)}T00:00:00`);
  if (Number.isNaN(d.getTime())) return false;
  const start = new Date(d.getFullYear(), 5, 1);
  const end = new Date(d.getFullYear(), 7, 31);
  return d >= start && d <= end;
}

function riskClass(v) {
  if (v >= 75) return 'very-high';
  if (v >= 55) return 'high';
  if (v >= 35) return 'moderate';
  return 'low';
}

function prettyRisk(value) {
  return riskClass(value).replace('-', ' ').toUpperCase();
}

function localizedRisk(lang, value) {
  if (lang !== 'hi') return prettyRisk(value);
  const labels = {
    low: 'कम',
    moderate: 'मध्यम',
    high: 'उच्च',
    'very-high': 'बहुत उच्च',
  };
  return labels[riskClass(value)] || prettyRisk(value);
}

const SPATIAL_SNAPSHOT_URL =
  '/mausamspatial/step7b_spatial_map_snapshot_20230901.geojson';
const SPATIAL_API_URL = `${API}/ml-spatial-forecast`;
const SPATIAL_PREDICTION_DATE = '2025-09-01';


const BLOCK_BOUNDARY_DEFS = {
  Belha: { color: '#2563EB', dash: '10 6' },
  Kota: { color: '#7C3AED', dash: '5 5' },
  Masturi: { color: '#0F766E', dash: '12 4 3 4' },
  Takhatpur: { color: '#334155', dash: '2 4' },
};

const MAP_LAYER_DEFS = {
  heavyRainPanchayat: {
    base: 'heavy_rain_panchayat_mean',
    en: 'Heavy Rain Risk',
    hi: 'भारी वर्षा जोखिम',
  },
  onset: {
    base: 'onset',
    en: 'Monsoon Onset',
    hi: 'मानसून आगमन',
  },
  breakProxy: {
    base: 'break_proxy',
    en: 'Dry Spell Risk',
    hi: 'शुष्क अवधि जोखिम',
  },
};

function mapTargetName(horizonDays, mapLayer) {
  return `${MAP_LAYER_DEFS[mapLayer].base}_${horizonDays}d`;
}

function mapHeading(lang, mapLayer) {
  const labels = {
    heavyRainPanchayat: {
      en: 'Heavy Rain Risk',
      hi: 'भारी वर्षा जोखिम',
    },
    onset: {
      en: 'Monsoon Onset',
      hi: 'मानसून आगमन',
    },
    breakProxy: {
      en: 'Dry Spell Risk',
      hi: 'शुष्क अवधि जोखिम',
    },
  };

  return labels[mapLayer]?.[lang] || labels.heavyRainPanchayat[lang];
}

function collectGeoJsonPositions(value, out = []) {
  if (!Array.isArray(value)) return out;
  if (
    value.length >= 2 &&
    typeof value[0] === 'number' &&
    typeof value[1] === 'number'
  ) {
    out.push([value[0], value[1]]);
    return out;
  }

  value.forEach((child) => collectGeoJsonPositions(child, out));
  return out;
}

function buildGeoJsonProjector(geojson) {
  const positions = [];

  (geojson?.features || []).forEach((feature) => {
    collectGeoJsonPositions(feature?.geometry?.coordinates, positions);
  });

  if (positions.length === 0) return null;

  const minLon = Math.min(...positions.map((p) => p[0]));
  const maxLon = Math.max(...positions.map((p) => p[0]));
  const minLat = Math.min(...positions.map((p) => p[1]));
  const maxLat = Math.max(...positions.map((p) => p[1]));

  const width = 900;
  const height = 560;
  const padding = 22;

  const lonSpan = Math.max(maxLon - minLon, 1e-9);
  const latSpan = Math.max(maxLat - minLat, 1e-9);

  // Preserve geographic aspect ratio so the real boundary shapes are not
  // stretched independently in x/y.
  const scale = Math.min(
    (width - 2 * padding) / lonSpan,
    (height - 2 * padding) / latSpan
  );

  const contentWidth = lonSpan * scale;
  const contentHeight = latSpan * scale;

  const offsetX = (width - contentWidth) / 2;
  const offsetY = (height - contentHeight) / 2;

  return {
    width,
    height,
    project([lon, lat]) {
      return [
        offsetX + (lon - minLon) * scale,
        offsetY + (maxLat - lat) * scale,
      ];
    },
  };
}

function ringToPath(ring, project) {
  if (!Array.isArray(ring) || ring.length < 2) return '';

  return (
    ring
      .map((coord, index) => {
        const [x, y] = project(coord);
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
      })
      .join(' ') + ' Z'
  );
}

function geometryToSvgPath(geometry, project) {
  if (!geometry?.type || !geometry?.coordinates) return '';

  if (geometry.type === 'Polygon') {
    return geometry.coordinates.map((ring) => ringToPath(ring, project)).join(' ');
  }

  if (geometry.type === 'MultiPolygon') {
    return geometry.coordinates
      .map((polygon) =>
        polygon.map((ring) => ringToPath(ring, project)).join(' ')
      )
      .join(' ');
  }

  return '';
}

function geometryToOuterRings(geometry) {
  if (!geometry?.type || !geometry?.coordinates) return [];

  if (geometry.type === 'Polygon') {
    return geometry.coordinates?.[0] ? [geometry.coordinates[0]] : [];
  }

  if (geometry.type === 'MultiPolygon') {
    return geometry.coordinates
      .map((polygon) => polygon?.[0])
      .filter((ring) => Array.isArray(ring) && ring.length >= 2);
  }

  return [];
}

function roundedCoordinate(value) {
  return Math.round(Number(value) * 1e7) / 1e7;
}

function segmentKey(a, b) {
  const aKey = `${roundedCoordinate(a[0])},${roundedCoordinate(a[1])}`;
  const bKey = `${roundedCoordinate(b[0])},${roundedCoordinate(b[1])}`;
  return aKey < bKey ? `${aKey}|${bKey}` : `${bKey}|${aKey}`;
}

function buildBlockBoundaryPaths(geojson, blockByLocalBodyCode, projector) {
  if (!geojson || !projector || !blockByLocalBodyCode) {
    return { paths: [], audit: [] };
  }

  const counters = new Map();

  (geojson.features || []).forEach((feature) => {
    const id = feature?.properties?.mausam_local_body_code;
    const blockName = blockByLocalBodyCode.get(String(id));

    if (!BLOCK_BOUNDARY_DEFS[blockName]) return;

    const rings = geometryToOuterRings(feature?.geometry);
    if (rings.length === 0) return;

    let blockCounter = counters.get(blockName);
    if (!blockCounter) {
      blockCounter = new Map();
      counters.set(blockName, blockCounter);
    }

    rings.forEach((ring) => {
      for (let index = 0; index < ring.length - 1; index += 1) {
        const a = ring[index];
        const b = ring[index + 1];
        if (!Array.isArray(a) || !Array.isArray(b)) continue;
        if (a.length < 2 || b.length < 2) continue;

        const key = segmentKey(a, b);
        const existing = blockCounter.get(key);
        if (existing) {
          existing.count += 1;
        } else {
          blockCounter.set(key, { a, b, count: 1 });
        }
      }
    });
  });

  const paths = [];
  const audit = [];

  Object.keys(BLOCK_BOUNDARY_DEFS).forEach((blockName) => {
    const counter = counters.get(blockName) || new Map();
    const uniqueSegments = [...counter.values()].filter((segment) => segment.count === 1);

    const path = uniqueSegments
      .map((segment) => {
        const [ax, ay] = projector.project(segment.a);
        const [bx, by] = projector.project(segment.b);
        return `M ${ax.toFixed(2)} ${ay.toFixed(2)} L ${bx.toFixed(2)} ${by.toFixed(2)}`;
      })
      .join(' ');

    paths.push({
      blockName,
      color: BLOCK_BOUNDARY_DEFS[blockName].color,
      path,
    });
  });

  return { paths, audit };
}

const RISK_MAP_COLORS = {
  low: '#43A047',
  moderate: '#F2C94C',
  high: '#F2994A',
  'very-high': '#D64545',
};

const RISK_MAP_STOPS = [
  [0, '#15803D'],
  [25, '#65A30D'],
  [50, '#FACC15'],
  [75, '#F97316'],
  [100, '#DC2626'],
];

function probabilityToColor(percent) {
  const value = Math.max(0, Math.min(100, Number(percent) || 0));

  for (let index = 0; index < RISK_MAP_STOPS.length - 1; index += 1) {
    const [leftValue, leftColor] = RISK_MAP_STOPS[index];
    const [rightValue, rightColor] = RISK_MAP_STOPS[index + 1];

    if (value <= rightValue) {
      const ratio = (value - leftValue) / (rightValue - leftValue);
      const leftRgb = leftColor.match(/^#([0-9a-f]{6})$/i)[1]
        .match(/../g)
        .map((part) => parseInt(part, 16));
      const rightRgb = rightColor.match(/^#([0-9a-f]{6})$/i)[1]
        .match(/../g)
        .map((part) => parseInt(part, 16));

      const rgb = leftRgb.map((channel, channelIndex) =>
        Math.round(channel + (rightRgb[channelIndex] - channel) * ratio)
      );

      return `rgb(${rgb[0]} ${rgb[1]} ${rgb[2]})`;
    }
  }

  return RISK_MAP_STOPS[RISK_MAP_STOPS.length - 1][1];
}

function formatProbability(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return '—';
  return `${number.toFixed(1)}%`;
}

function App() {

  useEffect(() => {
    if (!supabase) {
      setAuthMessage(
        lang === 'hi'
          ? 'Supabase frontend configuration उपलब्ध नहीं है।'
          : 'Supabase frontend configuration is missing.'
      );
      return undefined;
    }

    let active = true;

    supabase.auth.getSession().then(({ data, error }) => {
      if (!active) return;

      if (error) {
        setAuthMessage(error.message);
        return;
      }

      const session = data.session || null;
      setAuthSession(session);
      setAuthUser(session?.user || null);

      if (session) {
        window.setTimeout(() => bootstrapAuthenticatedSession(session), 0);
      }
    });

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      if (!active) return;

      setAuthSession(session || null);
      setAuthUser(session?.user || null);

      if (session) {
        window.setTimeout(() => bootstrapAuthenticatedSession(session), 0);
      }
    });

    return () => {
      active = false;
      subscription.unsubscribe();
    };
  }, []);

  async function backendAuthPost(path, body = {}) {
    if (!supabase) {
      throw new Error(
        lang === 'hi'
          ? 'Supabase frontend configuration उपलब्ध नहीं है।'
          : 'Supabase frontend configuration is missing.'
      );
    }

    const { data, error } = await supabase.auth.getSession();

    if (error || !data.session?.access_token) {
      throw new Error(
        lang === 'hi' ? 'पहले साइन इन करें।' : 'Please sign in first.'
      );
    }

    const response = await fetch(`${API}${path}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${data.session.access_token}`,
      },
      body: JSON.stringify(body),
    });

    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      throw new Error(payload.detail || 'Authenticated request failed.');
    }

    return payload;
  }

  async function bootstrapAuthenticatedSession(session) {
    if (!session?.access_token) return;

    try {
      const response = await fetch(`${API}/persistence/me/bootstrap`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${session.access_token}`,
        },
      });

      const payload = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(payload.detail || 'Account bootstrap failed.');
      }

      let deviceId = window.localStorage.getItem('mausamsaathi_device_id');

      if (!deviceId) {
        deviceId = window.crypto?.randomUUID
          ? window.crypto.randomUUID()
          : `browser-${Date.now()}-${Math.random().toString(36).slice(2)}`;
        window.localStorage.setItem('mausamsaathi_device_id', deviceId);
      }

      const deviceResponse = await fetch(`${API}/persistence/me/devices`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${session.access_token}`,
        },
        body: JSON.stringify({
          device_id: deviceId,
          platform: 'web',
          app_version: 'step19',
        }),
      });

      if (!deviceResponse.ok) {
        const devicePayload = await deviceResponse.json().catch(() => ({}));
        throw new Error(devicePayload.detail || 'Device bootstrap failed.');
      }

      setAuthMessage(
        lang === 'hi'
          ? 'अकाउंट जुड़ा है और लोकल/क्लाउड persistence तैयार है।'
          : 'Account connected and local/cloud persistence is ready.'
      );
    } catch (error) {
      setAuthMessage(error.message || 'Account bootstrap failed.');
    }
  }

  async function handleAuthSubmit(event) {
    event.preventDefault();

    if (!supabase) {
      setAuthMessage(
        lang === 'hi'
          ? 'Supabase frontend configuration उपलब्ध नहीं है।'
          : 'Supabase frontend configuration is missing.'
      );
      return;
    }

    if (!authEmail.trim() || !authPassword) {
      setAuthMessage(
        lang === 'hi'
          ? 'ईमेल और पासवर्ड दर्ज करें।'
          : 'Enter email and password.'
      );
      return;
    }

    setAuthBusy(true);
    setAuthMessage('');

    try {
      if (authMode === 'signup') {
        const { data, error } = await supabase.auth.signUp({
          email: authEmail.trim(),
          password: authPassword,
          options: {
            data: {
              preferred_language: lang,
            },
          },
        });

        if (error) throw error;

        if (data.session) {
          await bootstrapAuthenticatedSession(data.session);
        } else {
          setAuthMessage(
            lang === 'hi'
              ? 'अकाउंट बन गया। यदि email confirmation चालू है तो ईमेल देखें, फिर साइन इन करें।'
              : 'Account created. Check your email if confirmation is enabled, then sign in.'
          );
        }
      } else {
        const { data, error } = await supabase.auth.signInWithPassword({
          email: authEmail.trim(),
          password: authPassword,
        });

        if (error) throw error;

        setAuthSession(data.session);
        setAuthUser(data.user);
        await bootstrapAuthenticatedSession(data.session);
      }

      setAuthPassword('');
    } catch (error) {
      setAuthMessage(error.message || 'Authentication failed.');
    } finally {
      setAuthBusy(false);
    }
  }

  async function handleSignOut() {
    if (!supabase) return;

    setAuthBusy(true);

    try {
      const { error } = await supabase.auth.signOut();
      if (error) throw error;

      setAuthSession(null);
      setAuthUser(null);
      setAuthMessage(lang === 'hi' ? 'साइन आउट हो गया।' : 'Signed out.');
    } catch (error) {
      setAuthMessage(error.message || 'Could not sign out.');
    } finally {
      setAuthBusy(false);
    }
  }
  const [locations, setLocations] = useState(null);

  const [state, setState] = useState('');
  const [district, setDistrict] = useState('');
  const [block, setBlock] = useState('');
  const [panchayat, setPanchayat] = useState('');

  const [horizon, setHorizon] = useState(14);
  const [forecast, setForecast] = useState(null);

  const [crop, setCrop] = useState('rice');
  const [lang, setLang] = useState('en');
  const [advisory, setAdvisory] = useState(null);

  const [selectedArea, setSelectedArea] = useState(null);
  const [spatialMap, setSpatialMap] = useState(null);
  const [spatialMapError, setSpatialMapError] = useState('');
  const [spatialMapProbabilities, setSpatialMapProbabilities] = useState(null);
  const [spatialMapPredictionDate, setSpatialMapPredictionDate] = useState('');
  const [spatialMapPredictionLoading, setSpatialMapPredictionLoading] = useState(false);
  const [mapLayer, setMapLayer] = useState('heavyRainPanchayat');
  const [mapProbabilityType] = useState('calibrated');
  const [mapScope, setMapScope] = useState('selected');
  const [selectedMlSummary, setSelectedMlSummary] = useState(null);

  const [toast, setToast] = useState('');
  const [deliveryStatus, setDeliveryStatus] = useState('');
  const [deliveryChannel, setDeliveryChannel] = useState('');
  const [availableVoices, setAvailableVoices] = useState([]);
  const [sosStatus, setSosStatus] = useState('');
  const [deviceLocationBusy, setDeviceLocationBusy] = useState(false);
  const [deviceLocationMessage, setDeviceLocationMessage] = useState('');
  const [authSession, setAuthSession] = useState(null);
  const [authUser, setAuthUser] = useState(null);
  const [authMode, setAuthMode] = useState('signin');
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');
  const [authBusy, setAuthBusy] = useState(false);
  const [authMessage, setAuthMessage] = useState('');

  useEffect(() => {
    fetch(`${API}/locations`)
      .then((r) => r.json())
      .then(setLocations)
      .catch(() => {
        showToast('Backend not running. Start FastAPI first.');
      });
  }, []);

  useEffect(() => {
    if (!('speechSynthesis' in window)) return;

    const refreshVoices = () => {
      setAvailableVoices(window.speechSynthesis.getVoices());
    };

    refreshVoices();
    window.speechSynthesis.addEventListener('voiceschanged', refreshVoices);

    return () => {
      window.speechSynthesis.removeEventListener('voiceschanged', refreshVoices);
    };
  }, []);

  useEffect(() => {
    fetch(SPATIAL_SNAPSHOT_URL)
      .then((response) => {
        if (!response.ok) {
          throw new Error('Spatial boundary snapshot not found.');
        }
        return response.json();
      })
      .then((data) => {
        if (data?.type !== 'FeatureCollection' || !Array.isArray(data.features)) {
          throw new Error('Spatial boundary snapshot has an invalid GeoJSON structure.');
        }
        if (data.features.length !== 486) {
          throw new Error(
            `Expected 486 spatial features, found ${data.features.length}.`
          );
        }

        setSpatialMap(data);
        setSpatialMapError('');
      })
      .catch((error) => {
        setSpatialMap(null);
        setSpatialMapError(error.message || 'Could not load spatial boundaries.');
      });
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const indicator = MAP_LAYER_DEFS[mapLayer]?.base;

    if (!indicator) {
      setSpatialMapProbabilities(null);
      setSpatialMapPredictionDate('');
      setSpatialMapPredictionLoading(false);
      return () => controller.abort();
    }

    setSpatialMapPredictionLoading(true);
    setSpatialMapError('');

    const params = new URLSearchParams({
      horizon: String(horizon),
      indicator,
      probability_type: mapProbabilityType,
    });

    fetch(`${SPATIAL_API_URL}?${params.toString()}`, {
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) {
          return response
            .json()
            .catch(() => ({}))
            .then((data) => {
              throw new Error(data.detail || 'Spatial ML prediction request failed.');
            });
        }
        return response.json();
      })
      .then((data) => {
        if (data?.status !== 'ok') {
          throw new Error('Spatial ML API returned an invalid response.');
        }
        if (data.local_body_count !== 486 || !Array.isArray(data.predictions)) {
          throw new Error('Spatial ML API did not return all 486 local bodies.');
        }

        const probabilityMap = new Map(
          data.predictions.map((item) => [String(item.local_body_code), item])
        );

        if (probabilityMap.size !== 486) {
          throw new Error('Spatial ML API returned duplicate or missing local bodies.');
        }

        setSpatialMapProbabilities(probabilityMap);
        setSpatialMapPredictionDate(data.snapshot_date || '');
        setSpatialMapPredictionLoading(false);
        setSpatialMapError('');
      })
      .catch((error) => {
        if (error.name === 'AbortError') return;
        setSpatialMapProbabilities(null);
        setSpatialMapPredictionDate('');
        setSpatialMapPredictionLoading(false);
        setSpatialMapError(error.message || 'Could not load spatial ML predictions.');
      });

    return () => controller.abort();
  }, [horizon, mapLayer, mapProbabilityType]);

  // Fetch all ML indicators for the currently selected Panchayat.
  // This keeps the probabilistic outlook populated without requiring a map click.
  // A map click may still update `selectedArea` independently for map-detail display.
  useEffect(() => {
    const localBodyCode = panchayat || selectedArea?.id;

    if (!localBodyCode) {
      setSelectedMlSummary(null);
      return undefined;
    }

    const controller = new AbortController();

    const params = new URLSearchParams({
      horizon: String(horizon),
      probability_type: mapProbabilityType,
      local_body_code: String(localBodyCode),
    });

    fetch(`${SPATIAL_API_URL}?${params.toString()}`, {
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) {
          return response
            .json()
            .catch(() => ({}))
            .then((data) => {
              throw new Error(
                data.detail || 'Selected Panchayat ML request failed.'
              );
            });
        }
        return response.json();
      })
      .then((data) => {
        if (
          data?.status !== 'ok' ||
          data.local_body_count !== 1 ||
          !data.indicators
        ) {
          throw new Error('Selected Panchayat ML response is invalid.');
        }

        const getProbability = (indicator) => {
          const value = data.indicators[indicator]?.[0]?.probability;
          return Number.isFinite(Number(value)) ? Number(value) : null;
        };

        setSelectedMlSummary({
          onset: getProbability('onset'),
          breakProxy: getProbability('break_proxy'),
          heavyRain: getProbability('heavy_rain_panchayat_mean'),
          anomalyMm: Number.isFinite(Number(data.rainfall_anomaly_mm))
            ? Number(data.rainfall_anomaly_mm)
            : null,
        });
      })
      .catch((error) => {
        if (error.name === 'AbortError') return;
        setSelectedMlSummary(null);
        console.error(error);
      });

    return () => controller.abort();
  }, [panchayat, selectedArea?.id, horizon, mapProbabilityType]);

  const stateObj = locations?.states?.find((s) => s.name === state);
  const districtObj = stateObj?.districts?.find((d) => d.name === district);
  const blockObj = districtObj?.blocks?.find((b) => b.name === block);


  const blockByLocalBodyCode = useMemo(() => {
    const mapping = new Map();

    locations?.states?.forEach((stateItem) => {
      stateItem?.districts?.forEach((districtItem) => {
        districtItem?.blocks?.forEach((blockItem) => {
          blockItem?.panchayats?.forEach((panchayatItem) => {
            if (panchayatItem?.local_body_code != null) {
              mapping.set(
                String(panchayatItem.local_body_code),
                blockItem.name
              );
            }
          });
        });
      });
    });

    return mapping;
  }, [locations]);

  useEffect(() => {
    setDistrict('');
    setBlock('');
    setPanchayat('');
    setForecast(null);
    setAdvisory(null);
    setSelectedArea(null);
    setDeliveryStatus('');
    setSosStatus('');
  }, [state]);

  useEffect(() => {
    setBlock('');
    setPanchayat('');
    setForecast(null);
    setAdvisory(null);
    setSelectedArea(null);
    setDeliveryStatus('');
    setSosStatus('');
  }, [district]);

  useEffect(() => {
    setPanchayat('');
    setForecast(null);
    setAdvisory(null);
    setSelectedArea(null);
    setDeliveryStatus('');
    setSosStatus('');
  }, [block]);

  const selectedPanchayatObj = useMemo(() => {
    if (!blockObj?.panchayats || !panchayat) return null;
    return blockObj.panchayats.find(
      (item) => String(item.local_body_code) === String(panchayat)
    ) || null;
  }, [blockObj, panchayat]);

  const locationId = selectedPanchayatObj?.location_id || '';

  function showToast(message) {
    setToast(message);

    window.clearTimeout(window.__msToastTimer);

    window.__msToastTimer = window.setTimeout(() => {
      setToast('');
    }, 2500);
  }


  async function handleUseMyLocation() {
    if (!locations) {
      const message = lang === 'hi'
        ? 'स्थान सूची अभी लोड नहीं हुई है।'
        : 'Location list is still loading.';
      setDeviceLocationMessage(message);
      showToast(message);
      return;
    }

    if (!spatialMap) {
      const message = lang === 'hi'
        ? 'पंचायत सीमाएँ अभी लोड नहीं हुई हैं।'
        : 'Panchayat boundaries are still loading.';
      setDeviceLocationMessage(message);
      showToast(message);
      return;
    }

    if (!navigator.geolocation) {
      const message = lang === 'hi'
        ? 'इस डिवाइस/ब्राउज़र में लोकेशन सेवा उपलब्ध नहीं है। मैन्युअल चयन करें।'
        : 'Location services are not available in this browser. Please use manual selection.';
      setDeviceLocationMessage(message);
      showToast(message);
      return;
    }

    setDeviceLocationBusy(true);
    setDeviceLocationMessage(
      lang === 'hi'
        ? 'आपकी डिवाइस लोकेशन खोजी जा रही है…'
        : 'Finding your device location…'
    );

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        try {
          const latitude = Number(position.coords.latitude);
          const longitude = Number(position.coords.longitude);
          const accuracy = Number(position.coords.accuracy);

          if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
            throw new Error(
              lang === 'hi'
                ? 'डिवाइस ने मान्य GPS coordinates नहीं दिए।'
                : 'The device did not return valid GPS coordinates.'
            );
          }

          const feature =
            (spatialMap.features || []).find((candidate) =>
              pointInFeature(longitude, latitude, candidate)
            ) || null;

          if (!feature) {
            throw new Error(
              lang === 'hi'
                ? 'आपकी लोकेशन 486 उपलब्ध पंचायत सीमाओं के अंदर नहीं मिली। मैन्युअल चयन का उपयोग करें।'
                : 'Your location was not found inside the 486 available Panchayat boundaries. Please use manual selection.'
            );
          }

          const localBodyCode = extractFeatureLocalBodyCode(feature);
          const record = findLocationRecordByLocalBodyCode(locations, localBodyCode);

          if (!record?.locationId) {
            throw new Error(
              lang === 'hi'
                ? 'लोकेशन मिली, लेकिन आधिकारिक Local Body Code से पंचायत रिकॉर्ड नहीं मिला।'
                : 'Location found, but no authoritative Panchayat record matched the Local Body Code.'
            );
          }

          // Existing selectors reset downstream values when parents change.
          // Apply the hierarchy in sequence so those resets settle naturally.
          setState(record.stateName);
          window.setTimeout(() => setDistrict(record.districtName), 60);
          window.setTimeout(() => setBlock(record.blockName), 120);
          window.setTimeout(() => {
            setPanchayat(record.panchayatCode);
            setMapScope('selected');
          }, 180);

          window.localStorage.setItem(
            'mausamsaathi_last_local_body_code',
            record.panchayatCode
          );

          if (authSession) {
            backendAuthPost('/persistence/me/locations', {
              local_body_code: record.panchayatCode,
              is_primary: true,
            }).catch((error) => {
              setAuthMessage(error.message || 'Could not sync detected location.');
            });

            backendAuthPost('/persistence/me/crops', {
              local_body_code: record.panchayatCode,
              crop_name: crop,
            }).catch((error) => {
              setAuthMessage(error.message || 'Could not sync crop profile.');
            });
          }

          const accuracyText = Number.isFinite(accuracy)
            ? `${Math.round(accuracy)} m`
            : '';

          setDeviceLocationMessage(
            lang === 'hi'
              ? `लोकेशन मिली: ${record.panchayatName}, ${record.blockName}. ${accuracyText ? `डिवाइस सटीकता लगभग ${accuracyText}. ` : ''}अब पूर्वानुमान देखें।`
              : `Location found: ${record.panchayatName}, ${record.blockName}. ${accuracyText ? `Device accuracy about ${accuracyText}. ` : ''}Now view the forecast.`
          );

          showToast(
            lang === 'hi'
              ? `${record.panchayatName} पंचायत अपने आप चुन ली गई।`
              : `${record.panchayatName} Panchayat selected automatically.`
          );
        } catch (error) {
          const message =
            error instanceof Error
              ? error.message
              : lang === 'hi'
                ? 'लोकेशन से पंचायत नहीं मिली।'
                : 'Could not match your location to a Panchayat.';
          setDeviceLocationMessage(message);
          showToast(message);
        } finally {
          setDeviceLocationBusy(false);
        }
      },
      (error) => {
        const messages = {
          1: lang === 'hi'
            ? 'लोकेशन अनुमति नहीं मिली। ब्राउज़र की location permission चालू करें या मैन्युअल चयन करें।'
            : 'Location permission was denied. Allow location access or use manual selection.',
          2: lang === 'hi'
            ? 'डिवाइस की लोकेशन उपलब्ध नहीं हो सकी।'
            : 'Device location is currently unavailable.',
          3: lang === 'hi'
            ? 'लोकेशन खोजने में समय समाप्त हो गया।'
            : 'Location lookup timed out.',
        };

        const message =
          messages[error.code] ||
          (lang === 'hi'
            ? 'डिवाइस लोकेशन प्राप्त नहीं हो सकी।'
            : 'Could not obtain device location.');

        setDeviceLocationMessage(message);
        setDeviceLocationBusy(false);
        showToast(message);
      },
      {
        enableHighAccuracy: true,
        timeout: 12000,
        maximumAge: 300000,
      }
    );
  }

  async function loadForecast() {
    if (!locationId) {
      showToast('Please select State → District → Block → Panchayat.');
      return;
    }

    try {
      const response = await fetch(
        `${API}/forecast/${locationId}?horizon=${horizon}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : 'Could not load forecast.'
        );
      }

      setForecast(data);
      setSelectedArea(null);
      setDeliveryStatus('');
      setDeliveryChannel('');
      setSosStatus('');

      if (authSession && data?.location?.local_body_code) {
        try {
          await backendAuthPost('/persistence/me/locations', {
            local_body_code: data.location.local_body_code,
            is_primary: true,
          });

          await backendAuthPost('/persistence/me/crops', {
            local_body_code: data.location.local_body_code,
            crop_name: crop,
          });
        } catch (error) {
          setAuthMessage(error.message || 'Could not sync account data.');
        }
      }
    } catch (error) {
      showToast(error.message || 'Could not load forecast.');
    }
  }

  async function loadAdvisory(
    language = lang,
    selectedCrop = crop,
    selectedHorizon = horizon
  ) {
    if (!locationId) return;

    try {
      const response = await fetch(
        `${API}/advisory/${locationId}/${selectedCrop}?horizon=${selectedHorizon}&lang=${language}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          typeof data.detail === 'string'
            ? data.detail
            : 'Could not load advisory.'
        );
      }

      setAdvisory(data.advisory);
    } catch (error) {
      showToast(error.message || 'Could not load advisory.');
    }
  }

  useEffect(() => {
    if (!locationId) return;

    loadAdvisory(lang, crop, horizon);
  }, [locationId, horizon, crop, lang]);

  function handleLanguageChange(language) {
    setLang(language);

    setDeliveryStatus('');
    setDeliveryChannel('');
    setSosStatus('');
  }

  function handleCropChange(nextCrop) {
    setCrop(nextCrop);

    if (authSession && forecast?.location?.local_body_code) {
      backendAuthPost('/persistence/me/crops', {
        local_body_code: forecast.location.local_body_code,
        crop_name: nextCrop,
      }).catch((error) => {
        setAuthMessage(error.message || 'Could not sync crop profile.');
      });
    }
  }

  function getAdvisoryIntro() {
    if (!advisory) return '';

    if (lang === 'hi') {
      const cropLabel = crop === 'rice' ? 'धान' : 'मक्का';
      return `चयनित ${horizon} दिनों के लिए ${cropLabel} की फसल-विशिष्ट कार्रवाई नीचे दी गई है।`;
    }

    const cropLabel = crop === 'rice' ? 'Rice' : 'Maize';
    return `Crop-specific actions for ${cropLabel} over the selected ${horizon}-day horizon are provided below.`;
  }

  function buildDeliveryMessage() {
    if (!forecast || !advisory) return '';

    const locationLabel = `${forecast.location.block} • ${forecast.location.panchayat}`;
    const deliveryHorizon = Number(horizon);
    const advisoryActions = Array.isArray(advisory.actions)
      ? advisory.actions.filter(Boolean)
      : [];

    if (lang === 'hi') {
      return (
        `${locationLabel} के लिए अगले ${deliveryHorizon} दिनों का मौसम संदेश। ` +
        `फसल सलाह: ${advisory.title || 'स्थानीय कृषि सलाह'}। ` +
        `${advisory.summary || 'स्थानीय कृषि सलाह देखें।'} ` +
        `${advisoryActions.length > 0 ? `अनुशंसित कार्रवाई: ${advisoryActions.join('। ')}` : ''}`
      ).trim();
    }

    return (
      `${locationLabel}: ${deliveryHorizon}-day weather advisory. ` +
      `Crop advisory: ${advisory.title || 'Local crop advisory'}. ` +
      `${advisory.summary || 'Check the local crop advisory.'} ` +
      `${advisoryActions.length > 0 ? `Recommended actions: ${advisoryActions.join('. ')}` : ''}`
    ).trim();
  }


  function handleDelivery(channel) {
    const message = buildDeliveryMessage();

    if (!message) {
      showToast('View a forecast first.');
      return;
    }

    setDeliveryChannel(channel);

    if (
      authSession &&
      forecast?.location?.local_body_code &&
      (channel === 'SMS' || channel === 'WhatsApp')
    ) {
      backendAuthPost('/persistence/me/notifications', {
        channel: channel.toLowerCase(),
        message,
        local_body_code: forecast.location.local_body_code,
        language: lang,
      }).catch((error) => {
        setAuthMessage(error.message || 'Could not queue notification.');
      });
    }

    if (channel === 'SMS') {
      setDeliveryStatus('SMS advisory prepared (demo mode)');
      showToast('SMS advisory prepared in selected language.');
      return;
    }

    if (channel === 'WhatsApp') {
      setDeliveryStatus('WhatsApp advisory prepared (demo mode)');
      showToast('WhatsApp advisory prepared.');
      return;
    }
  }

  function getCurrentDeliveryMetrics() {
    if (!forecast) return null;

    const forecastHorizonMatches =
      Number(forecast.horizon) === Number(horizon);

    const onset =
      selectedMlSummary?.onset != null
        ? Number(selectedMlSummary.onset) * 100
        : forecastHorizonMatches
          ? Number(forecast.forecast.onset_probability)
          : NaN;

    const breakProbability =
      selectedMlSummary?.breakProxy != null
        ? Number(selectedMlSummary.breakProxy) * 100
        : forecastHorizonMatches
          ? Number(forecast.forecast.break_probability)
          : NaN;

    const heavyRain =
      selectedMlSummary?.heavyRain != null
        ? Number(selectedMlSummary.heavyRain) * 100
        : forecastHorizonMatches
          ? Number(forecast.forecast.heavy_rain_probability)
          : NaN;

    if (![onset, breakProbability, heavyRain].every(Number.isFinite)) {
      return null;
    }

    return {
      horizon: Number(horizon),
      onset,
      breakProbability,
      heavyRain,
    };
  }

  function handleVoiceCall() {
    if (!forecast) {
      showToast('View a forecast first.');
      return;
    }

    if (!('speechSynthesis' in window)) {
      showToast('Voice playback is not supported in this browser.');
      return;
    }

    const deliveryMetrics = getCurrentDeliveryMetrics();

    if (!deliveryMetrics) {
      showToast('Selected horizon data is still loading. Please try again.');
      return;
    }

    const { onset, breakProbability, heavyRain } = deliveryMetrics;

    const advisoryActions = Array.isArray(advisory?.actions)
      ? advisory.actions.filter(Boolean)
      : [];

    // Voice-only Hindi location names. The visible website values remain unchanged.
    const hindiLocationNames = {
      Belha: 'बेल्हा',
      Kota: 'कोटा',
      Masturi: 'मस्तूरी',
      Takhatpur: 'तखतपुर',
      Changori: 'चांगोरी',
    };

    const hindiNumbers = {
      0: 'शून्य',
      1: 'एक',
      2: 'दो',
      3: 'तीन',
      4: 'चार',
      5: 'पाँच',
      6: 'छह',
      7: 'सात',
      8: 'आठ',
      9: 'नौ',
      10: 'दस',
      11: 'ग्यारह',
      12: 'बारह',
      13: 'तेरह',
      14: 'चौदह',
      15: 'पंद्रह',
      16: 'सोलह',
      17: 'सत्रह',
      18: 'अठारह',
      19: 'उन्नीस',
      20: 'बीस',
      21: 'इक्कीस',
      22: 'बाईस',
      23: 'तेईस',
      24: 'चौबीस',
      25: 'पच्चीस',
      26: 'छब्बीस',
      27: 'सत्ताईस',
      28: 'अट्ठाईस',
      29: 'उनतीस',
      30: 'तीस',
      31: 'इकतीस',
      32: 'बत्तीस',
      33: 'तैंतीस',
      34: 'चौंतीस',
      35: 'पैंतीस',
      36: 'छत्तीस',
      37: 'सैंतीस',
      38: 'अड़तीस',
      39: 'उनतालीस',
      40: 'चालीस',
      41: 'इकतालीस',
      42: 'बयालीस',
      43: 'तैंतालीस',
      44: 'चवालीस',
      45: 'पैंतालीस',
      46: 'छियालीस',
      47: 'सैंतालीस',
      48: 'अड़तालीस',
      49: 'उनचास',
      50: 'पचास',
      51: 'इक्यावन',
      52: 'बावन',
      53: 'तिरपन',
      54: 'चौवन',
      55: 'पचपन',
      56: 'छप्पन',
      57: 'सत्तावन',
      58: 'अट्ठावन',
      59: 'उनसठ',
      60: 'साठ',
      61: 'इकसठ',
      62: 'बासठ',
      63: 'तिरसठ',
      64: 'चौंसठ',
      65: 'पैंसठ',
      66: 'छियासठ',
      67: 'सड़सठ',
      68: 'अड़सठ',
      69: 'उनहत्तर',
      70: 'सत्तर',
      71: 'इकहत्तर',
      72: 'बहत्तर',
      73: 'तिहत्तर',
      74: 'चौहत्तर',
      75: 'पचहत्तर',
      76: 'छिहत्तर',
      77: 'सतहत्तर',
      78: 'अठहत्तर',
      79: 'उन्नासी',
      80: 'अस्सी',
      81: 'इक्यासी',
      82: 'बयासी',
      83: 'तिरासी',
      84: 'चौरासी',
      85: 'पचासी',
      86: 'छियासी',
      87: 'सत्तासी',
      88: 'अट्ठासी',
      89: 'नवासी',
      90: 'नब्बे',
      91: 'इक्यानबे',
      92: 'बानबे',
      93: 'तिरानबे',
      94: 'चौरानबे',
      95: 'पंचानबे',
      96: 'छियानबे',
      97: 'सत्तानबे',
      98: 'अट्ठानबे',
      99: 'निन्यानबे',
      100: 'एक सौ',
    };

    const hindiDecimalDigits = {
      0: 'शून्य',
      1: 'एक',
      2: 'दो',
      3: 'तीन',
      4: 'चार',
      5: 'पाँच',
      6: 'छह',
      7: 'सात',
      8: 'आठ',
      9: 'नौ',
    };

    function hindiNumber(value) {
      const number = Number(value);

      if (!Number.isFinite(number)) return 'उपलब्ध नहीं';

      const rounded = Math.round(number * 10) / 10;
      const integerPart = Math.floor(Math.abs(rounded));
      const decimalDigit = Math.round((Math.abs(rounded) - integerPart) * 10);

      let result = hindiNumbers[integerPart] || String(integerPart);

      if (decimalDigit > 0) {
        result += ` दशमलव ${hindiDecimalDigits[decimalDigit]}`;
      }

      return rounded < 0 ? `माइनस ${result}` : result;
    }

    function hindiLocationName(name) {
      const value = String(name || '').trim();
      return hindiLocationNames[value] || value;
    }

    const hindiBlock = hindiLocationName(forecast.location.block);
    const hindiPanchayat = hindiLocationName(forecast.location.panchayat);
    const hindiHorizon = hindiNumber(horizon);
    const hindiOnset = hindiNumber(onset);
    const hindiBreak = hindiNumber(breakProbability);
    const hindiHeavyRain = hindiNumber(heavyRain);

    const message = buildDeliveryMessage();
    if (!message) {
      showToast('Selected advisory is still loading. Please try again.');
      return;
    }

    const actionMessage = advisoryActions.length > 0
      ? (lang === 'hi'
          ? `अनुशंसित कार्रवाई: ${advisoryActions.join('। ')}`
          : `Recommended actions: ${advisoryActions.join('. ')}`)
      : '';

    const targetLang = lang === 'hi' ? 'hi-IN' : 'en-IN';

    const speak = () => {
      const voices = availableVoices.length > 0
        ? availableVoices
        : window.speechSynthesis.getVoices();

      const prefix = targetLang.toLowerCase().split('-')[0];
      const targetVoice =
        voices.find((voice) => voice.lang?.toLowerCase() === targetLang.toLowerCase()) ||
        voices.find((voice) => voice.lang?.toLowerCase().startsWith(prefix));

      if (!targetVoice) {
        setDeliveryChannel('Voice Call');
        setDeliveryStatus(
          lang === 'hi'
            ? 'Hindi voice is not installed on this device'
            : 'English voice is not available on this device'
        );
        showToast(
          lang === 'hi'
            ? 'Hindi voice not available. Install Hindi Text-to-Speech in Windows.'
            : 'English voice not available on this device.'
        );
        return;
      }

      window.speechSynthesis.cancel();
      const sequenceId = (window.__msVoiceSequenceId || 0) + 1;
      window.__msVoiceSequenceId = sequenceId;

      const speakPart = (part, onEnd) => {
        const utterance = new SpeechSynthesisUtterance(part);
        utterance.voice = targetVoice;
        utterance.lang = targetVoice.lang;
        utterance.rate = lang === 'hi' ? 0.86 : 0.9;
        utterance.pitch = 1;
        utterance.onend = onEnd;
        window.speechSynthesis.speak(utterance);
      };

      const repeatCount = 3;
      const delayMs = 700;

      const speakAction = (index) => {
        if (sequenceId !== window.__msVoiceSequenceId || !actionMessage) return;

        speakPart(actionMessage, () => {
          if (sequenceId !== window.__msVoiceSequenceId) return;

          if (index < repeatCount - 1) {
            window.setTimeout(() => speakAction(index + 1), delayMs);
          }
        });
      };

      speakPart(message, () => {
        if (sequenceId !== window.__msVoiceSequenceId) return;

        if (actionMessage) {
          speakAction(0);
        }
      });

      setDeliveryChannel('Voice Call');
      setDeliveryStatus(
        lang === 'hi'
          ? `Hindi regional voice playing • ${targetVoice.name}`
          : `English voice playing • ${targetVoice.name}`
      );
      showToast(
        lang === 'hi'
          ? 'Hindi regional voice advisory playing.'
          : 'Voice advisory playing.'
      );
    };

    if (availableVoices.length === 0) {
      window.setTimeout(speak, 350);
    } else {
      speak();
    }
  }

  async function handleSos() {
    if (!forecast) {
      showToast('View a forecast first.');
      return;
    }

    const message =
      `SOS alert for ${forecast.location.block} • ${forecast.location.panchayat}`;

    if (authSession) {
      try {
        await backendAuthPost('/persistence/me/sos', {
          local_body_code: forecast.location.local_body_code,
          payload: {
            source: 'web_dashboard',
            message,
            issue_date: forecast.forecast.issue_date,
            horizon,
          },
        });

        setSosStatus(
          lang === 'hi'
            ? `SOS स्थानीय persistence में दर्ज हो गया • ${forecast.location.block} • ${forecast.location.panchayat}`
            : `SOS persisted locally • ${forecast.location.block} • ${forecast.location.panchayat}`
        );
        showToast('SOS event saved to local persistence.');
        return;
      } catch (error) {
        setAuthMessage(error.message || 'Could not persist SOS event.');
      }
    }

    setSosStatus(
      `${message} (demo mode)`
    );
    showToast('SOS alert prepared for local response team.');
  }

  function handleSupportCall() {
    if (!forecast) {
      showToast('View a forecast first.');
      return;
    }

    setSosStatus(
      `Support call flow initiated for ${forecast.location.block} • ${forecast.location.panchayat} (demo mode)`
    );
    showToast('Support-call flow initiated in demo mode.');
  }

  const mapDisplayGeoJson = useMemo(() => {
    if (!spatialMap) return null;

    if (mapScope === 'selected') {
      return {
        ...spatialMap,
        features: spatialMap.features.filter((feature) => {
          const id = feature?.properties?.mausam_local_body_code;

          if (panchayat) {
            return String(id) === String(panchayat);
          }

          return blockByLocalBodyCode.get(String(id)) === block;
        }),
      };
    }

    return spatialMap;
  }, [spatialMap, mapScope, panchayat, block, blockByLocalBodyCode]);

  const mapProjector = useMemo(
    () => buildGeoJsonProjector(mapDisplayGeoJson),
    [mapDisplayGeoJson]
  );


  const blockBoundaryAudit = useMemo(() => {
    if (mapScope !== 'all' || !spatialMap || !locations) {
      return { valid: true, details: [] };
    }

    const expected = {
      Belha: 0,
      Kota: 0,
      Masturi: 0,
      Takhatpur: 0,
    };

    locations?.states?.forEach((stateItem) => {
      stateItem?.districts?.forEach((districtItem) => {
        districtItem?.blocks?.forEach((blockItem) => {
          if (Object.prototype.hasOwnProperty.call(expected, blockItem.name)) {
            expected[blockItem.name] = blockItem.panchayats?.length || 0;
          }
        });
      });
    });

    const actual = {
      Belha: 0,
      Kota: 0,
      Masturi: 0,
      Takhatpur: 0,
    };

    spatialMap.features.forEach((feature) => {
      const id = feature?.properties?.mausam_local_body_code;
      const blockName = blockByLocalBodyCode.get(String(id));
      if (Object.prototype.hasOwnProperty.call(actual, blockName)) {
        actual[blockName] += 1;
      }
    });

    const details = Object.keys(expected)
      .map((blockName) => ({
        blockName,
        expected: expected[blockName],
        actual: actual[blockName],
      }))
      .filter((item) => item.expected !== item.actual);

    return {
      valid: details.length === 0,
      details,
    };
  }, [mapScope, spatialMap, locations, blockByLocalBodyCode]);

  const blockBoundaryData = useMemo(() => {
    if (
      mapScope !== 'all' ||
      !spatialMap ||
      !mapProjector ||
      !blockBoundaryAudit.valid
    ) {
      return [];
    }

    return buildBlockBoundaryPaths(
      spatialMap,
      blockByLocalBodyCode,
      mapProjector
    ).paths;
  }, [
    mapScope,
    spatialMap,
    mapProjector,
    blockBoundaryAudit.valid,
    blockByLocalBodyCode,
  ]);

  const renderedMapFeatures = useMemo(() => {
    if (!spatialMap || !mapProjector || !spatialMapProbabilities) return [];

    return spatialMap.features.flatMap((feature) => {
      const props = feature.properties || {};
      const id = props.mausam_local_body_code;

      if (mapScope === 'selected' && String(id) !== String(panchayat)) {
        return [];
      }

      const apiPrediction = spatialMapProbabilities.get(String(id));

      if (!apiPrediction) return [];

      const probability = Number(apiPrediction.probability);

      if (!Number.isFinite(probability) || probability < 0 || probability > 1) {
        return [];
      }

      return [{
        id,
        label: apiPrediction.local_body_name || props.mausam_local_body_name,
        probability,
        path: geometryToSvgPath(feature.geometry, mapProjector.project),
      }];
    });
  }, [spatialMap, mapProjector, spatialMapProbabilities, mapScope, panchayat]);

  useEffect(() => {
    if (!selectedArea?.id || renderedMapFeatures.length === 0) return;

    const updated = renderedMapFeatures.find(
      (feature) => String(feature.id) === String(selectedArea.id)
    );

    if (!updated) return;

    setSelectedArea((current) => {
      if (
        current &&
        current.id === updated.id &&
        Math.abs(Number(current.risk) - Number(updated.probability) * 100) < 0.001
      ) {
        return current;
      }

      return {
        id: updated.id,
        label: updated.label,
        risk: updated.probability * 100,
      };
    });
  }, [renderedMapFeatures, selectedArea?.id]);

  function selectMapFeature(feature) {
    setSelectedArea({
      id: feature.id,
      label: feature.label,
      risk: feature.probability * 100,
    });
  }

  function getSelectedRiskValue() {
    // Prefer an explicitly clicked map feature. Otherwise, use the currently
    // selected Panchayat's probability so the card is populated immediately
    // without requiring a map click.
    if (selectedArea?.risk != null) return selectedArea.risk;

    if (panchayat && spatialMapProbabilities) {
      const prediction = spatialMapProbabilities.get(String(panchayat));
      const probability = Number(prediction?.probability);
      if (Number.isFinite(probability) && probability >= 0 && probability <= 1) {
        return probability * 100;
      }
    }

    return null;
  }

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark" aria-hidden="true">
            <span className="sun">☀</span>
            <span className="leaf">🌱</span>
          </div>
          <div>
            <div className="brand-hindi">मौसमसाथी</div>
            <div className="sub">{t(lang, 'tagline')}</div>
          </div>
        </div>

        <div className="header-tools">
          <label className="language-picker">
            <span>{t(lang, 'language')}</span>
            <select
              value={lang}
              onChange={(e) => handleLanguageChange(e.target.value)}
              aria-label="Dashboard language"
            >
              {LANGUAGES.map((language) => (
                <option key={language.code} value={language.code}>
                  {language.label}
                </option>
              ))}
            </select>
          </label>
          <div className="header-tag">🌾 {t(lang, 'headerTag')}</div>
        </div>
      </div>


      
      <details
        className="panel controls"
        style={{
          marginBottom: '18px',
          overflow: 'hidden',
          borderRadius: '18px',
        }}
      >
        <summary
          style={{
            cursor: 'pointer',
            listStyle: 'none',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: '16px',
            padding: '4px 0',
            fontWeight: 800,
          }}
        >
          <span
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
            }}
          >
            <span
              style={{
                width: '40px',
                height: '40px',
                borderRadius: '12px',
                display: 'grid',
                placeItems: 'center',
                background: 'rgba(24, 119, 180, 0.10)',
                fontSize: '1.2rem',
              }}
            >
              👤
            </span>

            <span>
              <span
                style={{
                  display: 'block',
                  fontSize: '1rem',
                  lineHeight: 1.2,
                }}
              >
                {lang === 'hi'
                  ? 'वैकल्पिक अकाउंट और अलर्ट'
                  : 'Optional account & alerts'}
              </span>

              <span
                style={{
                  display: 'block',
                  marginTop: '4px',
                  fontSize: '0.84rem',
                  fontWeight: 500,
                  opacity: 0.72,
                }}
              >
                {lang === 'hi'
                  ? 'पूर्वानुमान देखने के लिए साइन इन जरूरी नहीं है'
                  : 'No sign-in is needed to view forecasts'}
              </span>
            </span>
          </span>

          <span
            style={{
              padding: '6px 10px',
              borderRadius: '999px',
              fontSize: '0.76rem',
              fontWeight: 800,
              letterSpacing: '0.02em',
              background: 'rgba(24, 119, 180, 0.08)',
              whiteSpace: 'nowrap',
            }}
          >
            {lang === 'hi' ? 'वैकल्पिक' : 'OPTIONAL'}
          </span>
        </summary>

        <div
          style={{
            marginTop: '16px',
            paddingTop: '16px',
            borderTop: '1px solid rgba(30, 60, 80, 0.10)',
          }}
        >
          {authUser ? (
            <div
              style={{
                display: 'grid',
                gap: '14px',
              }}
            >
              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '16px',
                  flexWrap: 'wrap',
                }}
              >
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '12px',
                  }}
                >
                  <div
                    style={{
                      width: '46px',
                      height: '46px',
                      borderRadius: '50%',
                      display: 'grid',
                      placeItems: 'center',
                      background: 'rgba(24, 119, 180, 0.12)',
                      fontWeight: 800,
                      fontSize: '1.1rem',
                    }}
                  >
                    {(authUser.email || 'F').charAt(0).toUpperCase()}
                  </div>

                  <div>
                    <strong style={{ display: 'block', fontSize: '0.95rem' }}>
                      {lang === 'hi' ? 'अकाउंट सक्रिय' : 'Account active'}
                    </strong>

                    <span
                      style={{
                        display: 'block',
                        marginTop: '3px',
                        fontSize: '0.84rem',
                        opacity: 0.72,
                      }}
                    >
                      {authUser.email || authUser.phone || authUser.id}
                    </span>
                  </div>
                </div>

                <button
                  type="button"
                  className="secondary"
                  onClick={handleSignOut}
                  disabled={authBusy}
                  style={{ minWidth: '110px' }}
                >
                  {lang === 'hi' ? 'साइन आउट' : 'Sign out'}
                </button>
              </div>

              <div
                style={{
                  padding: '11px 13px',
                  borderRadius: '12px',
                  background: 'rgba(39, 150, 88, 0.08)',
                  fontSize: '0.86rem',
                }}
              >
                ✓ {lang === 'hi'
                  ? 'लोकल + क्लाउड persistence सक्रिय है'
                  : 'Local + cloud persistence is active'}
              </div>
            </div>
          ) : (
            <div style={{ display: 'grid', gap: '14px' }}>
              <div
                style={{
                  display: 'flex',
                  gap: '8px',
                  padding: '4px',
                  borderRadius: '12px',
                  background: 'rgba(30, 60, 80, 0.06)',
                  width: 'fit-content',
                  maxWidth: '100%',
                }}
              >
                <button
                  type="button"
                  onClick={() => {
                    setAuthMode('signin');
                    setAuthMessage('');
                  }}
                  style={{
                    border: 0,
                    borderRadius: '9px',
                    padding: '9px 14px',
                    fontWeight: 800,
                    background:
                      authMode === 'signin'
                        ? 'rgba(24, 119, 180, 0.14)'
                        : 'transparent',
                    cursor: 'pointer',
                  }}
                >
                  {lang === 'hi' ? 'साइन इन' : 'Sign in'}
                </button>

                <button
                  type="button"
                  onClick={() => {
                    setAuthMode('signup');
                    setAuthMessage('');
                  }}
                  style={{
                    border: 0,
                    borderRadius: '9px',
                    padding: '9px 14px',
                    fontWeight: 800,
                    background:
                      authMode === 'signup'
                        ? 'rgba(24, 119, 180, 0.14)'
                        : 'transparent',
                    cursor: 'pointer',
                  }}
                >
                  {lang === 'hi' ? 'खाता बनाएँ' : 'Create account'}
                </button>
              </div>

              <form
                onSubmit={handleAuthSubmit}
                style={{
                  display: 'grid',
                  gap: '12px',
                  maxWidth: '720px',
                }}
              >
                <div
                  style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))',
                    gap: '12px',
                  }}
                >
                  <label>
                    {lang === 'hi' ? 'ईमेल' : 'Email'}
                    <input
                      type="email"
                      value={authEmail}
                      onChange={(e) => setAuthEmail(e.target.value)}
                      placeholder="farmer@example.com"
                      autoComplete="email"
                    />
                  </label>

                  <label>
                    {lang === 'hi' ? 'पासवर्ड' : 'Password'}
                    <input
                      type="password"
                      value={authPassword}
                      onChange={(e) => setAuthPassword(e.target.value)}
                      placeholder={
                        authMode === 'signup'
                          ? 'Minimum 6 characters'
                          : 'Enter your password'
                      }
                      autoComplete={
                        authMode === 'signup'
                          ? 'new-password'
                          : 'current-password'
                      }
                    />
                  </label>
                </div>

                <div
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '10px',
                    flexWrap: 'wrap',
                  }}
                >
                  <button
                    type="submit"
                    disabled={authBusy}
                    className="primary"
                    style={{ minWidth: '160px' }}
                  >
                    {authMode === 'signup'
                      ? (lang === 'hi'
                          ? 'खाता बनाएँ'
                          : 'Create account')
                      : (lang === 'hi'
                          ? 'साइन इन'
                          : 'Sign in')}
                  </button>

                  <span
                    style={{
                      fontSize: '0.84rem',
                      opacity: 0.68,
                    }}
                  >
                    {lang === 'hi'
                      ? 'अकाउंट से सेव्ड लोकेशन और अलर्ट जैसी सुविधाएँ मिलेंगी।'
                      : 'An account enables saved location, alerts and sync.'}
                  </span>
                </div>
              </form>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                  gap: '10px',
                }}
              >
                <div
                  style={{
                    padding: '11px 12px',
                    borderRadius: '12px',
                    background: 'rgba(24, 119, 180, 0.06)',
                  }}
                >
                  <strong style={{ display: 'block', fontSize: '0.86rem' }}>
                    📍 {lang === 'hi' ? 'सेव्ड स्थान' : 'Saved location'}
                  </strong>
                  <span style={{ fontSize: '0.78rem', opacity: 0.7 }}>
                    {lang === 'hi'
                      ? 'अपनी पंचायत जल्दी खोलें'
                      : 'Open your Panchayat faster'}
                  </span>
                </div>

                <div
                  style={{
                    padding: '11px 12px',
                    borderRadius: '12px',
                    background: 'rgba(24, 119, 180, 0.06)',
                  }}
                >
                  <strong style={{ display: 'block', fontSize: '0.86rem' }}>
                    🔔 {lang === 'hi' ? 'अलर्ट' : 'Alerts'}
                  </strong>
                  <span style={{ fontSize: '0.78rem', opacity: 0.7 }}>
                    {lang === 'hi'
                      ? 'भविष्य में मौसम अलर्ट सिंक करें'
                      : 'Sync weather alerts later'}
                  </span>
                </div>
              </div>
            </div>
          )}

          {authMessage && (
            <div
              style={{
                marginTop: '12px',
                fontSize: '0.86rem',
                opacity: 0.84,
              }}
            >
              {authMessage}
            </div>
          )}
        </div>
      </details>


      <main>
        {/* HERO */}
        <section className="hero panel">
          <div>
            <div className="eyebrow">{t(lang, 'heroEyebrow')}</div>
            <h1>{t(lang, 'heroTitle')}</h1>
            <p>{t(lang, 'heroText')}</p>
          </div>

          <div className="work-flow">
            <div className="work-flow-title">{t(lang, 'workTitle')}</div>
            <div className="work-flow-row">
              <div className="flow-item"><span className="flow-icon">🌐</span><div><b>{t(lang, 'climate')}</b><small>ENSO • IOD • MJO</small></div></div>
              <b className="flow-arrow">→</b>
              <div className="flow-item"><span className="flow-icon">🌦</span><div><b>{t(lang, 'regional')}</b><small>{t(lang, 'rainTempHum')}</small></div></div>
              <b className="flow-arrow">→</b>
              <div className="flow-item"><span className="flow-icon">🧠</span><div><b>{t(lang, 'ai')}</b><small>{t(lang, 'largeLocal')}</small></div></div>
              <b className="flow-arrow">→</b>
              <div className="flow-item"><span className="flow-icon">📍</span><div><b>{t(lang, 'local')}</b><small>{t(lang, 'blockPanchayat')}</small></div></div>
            </div>
          </div>
        </section>

        {/* LOCATION */}
        <section className="panel controls">
          <div className="section-title">{t(lang, 'selectLocation')}</div>

          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '12px',
              flexWrap: 'wrap',
              margin: '10px 0 16px',
            }}
          >
            <button
              type="button"
              className="secondary"
              onClick={handleUseMyLocation}
              disabled={deviceLocationBusy || !locations || !spatialMap}
            >
              {deviceLocationBusy
                ? (lang === 'hi' ? '📍 लोकेशन खोज रहे हैं…' : '📍 Finding location…')
                : (lang === 'hi' ? '📍 मेरी लोकेशन उपयोग करें' : '📍 Use My Location')}
            </button>

            <span style={{ fontSize: '0.88rem', opacity: 0.78 }}>
              {lang === 'hi'
                ? 'GPS वैकल्पिक है। अनुमति न देने पर नीचे से पंचायत चुन सकते हैं।'
                : 'GPS is optional. You can continue with the manual selectors below.'}
            </span>
          </div>

          {deviceLocationMessage && (
            <div
              style={{
                marginBottom: '14px',
                fontSize: '0.92rem',
                opacity: 0.9,
              }}
            >
              {deviceLocationMessage}
            </div>
          )}

          <div className="grid4">
            <label>
              {t(lang, 'state')}
              <select
                value={state}
                onChange={(e) => setState(e.target.value)}
              >
                <option value="">{t(lang, 'selectState')}</option>

                {locations?.states?.map((s) => (
                  <option key={s.name} value={s.name}>
                    {s.name}
                  </option>
                ))}
              </select>
            </label>

            <label>
              {t(lang, 'district')}
              <select
                value={district}
                onChange={(e) => setDistrict(e.target.value)}
                disabled={!state}
              >
                <option value="">{t(lang, 'selectDistrict')}</option>

                {stateObj?.districts?.map((d) => (
                  <option key={d.name} value={d.name}>
                    {d.name}
                  </option>
                ))}
              </select>
            </label>

            <label>
              {t(lang, 'block')}
              <select
                value={block}
                onChange={(e) => setBlock(e.target.value)}
                disabled={!district}
              >
                <option value="">{t(lang, 'selectBlock')}</option>

                {districtObj?.blocks?.map((b) => (
                  <option key={b.name} value={b.name}>
                    {b.name}
                  </option>
                ))}
              </select>
            </label>

            <label>
              {t(lang, 'panchayat')}
              <select
                value={panchayat}
                onChange={(e) => setPanchayat(e.target.value)}
                disabled={!block}
              >
                <option value="">{t(lang, 'selectPanchayat')}</option>

                {blockObj?.panchayats?.map((p) => (
                  <option key={p.local_body_code} value={p.local_body_code}>
                    {p.name}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="horizon-row">
            <div className="section-title">{t(lang, 'horizon')}</div>

            <div className="horizon-buttons">
              {HORIZONS.map((h) => (
                <button
                  key={h}
                  className={horizon === h ? 'active' : ''}
                  onClick={() => {
                    setHorizon(h);
                  }}
                >
                  {h} {t(lang, 'days')}
                </button>
              ))}
            </div>

            <button className="primary" onClick={loadForecast}>
              {t(lang, 'viewForecast')}
            </button>
          </div>
        </section>

        {!forecast && (
          <section className="panel empty">
            <div className="empty-icon">🌦</div>
            <h2>{t(lang, 'readyTitle')}</h2>
            <p>
              {t(lang, 'readyText')}
            </p>
          </section>
        )}

        {forecast && (
          <>
            {/* FORECAST */}
            <section className="panel dashboard">
              <div className="dash-head">
                <div>
                  <div className="section-title">{t(lang, 'outlook')}</div>

                  <h2>
                    {forecast.location.block} •{' '}
                    {forecast.location.panchayat}
                  </h2>
                </div>

                <div className="confidence">
                  {t(lang, 'confidence')}
                  <strong>{forecast.forecast.confidence}</strong>
                  <span>{t(lang, 'demoView')}</span>
                </div>
              </div>

              <div className="monsoon-phase-strip" style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  flexWrap: 'wrap',
                  marginBottom: '14px',
                  padding: '10px 14px',
                  borderRadius: '12px',
                  border: '1px solid rgba(35, 83, 116, 0.14)',
                  background: 'rgba(255,255,255,0.72)',
                }}>
                <span>{lang === 'hi' ? 'मानसून चरण' : 'MONSOON PHASE'}</span>
                <strong>
                  {(() => {
                    const phaseDate =
                      spatialMapPredictionDate ||
                      forecast?.snapshot_date ||
                      forecast?.forecast?.snapshot_date ||
                      forecast?.forecast?.issue_date;
                    const phase = getMonsoonPhase(phaseDate);
                    return phase === 'PRE' ? 'PRE' : phase === 'MID' ? 'MID' : phase === 'POST' ? 'POST' : '—';
                  })()}
                </strong>
                <small>
                  {(() => {
                    const phaseDate =
                      spatialMapPredictionDate ||
                      forecast?.snapshot_date ||
                      forecast?.forecast?.snapshot_date ||
                      forecast?.forecast?.issue_date;
                    return isOnsetWindowOpen(phaseDate)
                      ? 'Onset probability applicable'
                      : 'Onset window closed';
                  })()}
                </small>
              </div>

              <div className="metrics">
                <Metric

                  title={t(lang, 'onset')}

                  value={(() => {

                    const phaseDate =

                      spatialMapPredictionDate ||

                      forecast?.snapshot_date ||

                      forecast?.forecast?.snapshot_date ||

                      forecast?.forecast?.issue_date;

                    if (!isOnsetWindowOpen(phaseDate)) return '—';

                    return selectedMlSummary?.onset != null

                      ? (selectedMlSummary.onset * 100).toFixed(1)

                      : forecast.forecast.onset_probability;

                  })()}

                  icon="🌧"

                />

                <Metric
                  title={t(lang, 'breakPhase')}
                  value={
                    selectedMlSummary?.breakProxy != null
                      ? (selectedMlSummary.breakProxy * 100).toFixed(1)
                      : forecast.forecast.break_probability
                  }
                  icon="☀"
                />

                <Metric
                  title={t(lang, 'heavyRain')}
                  value={
                    selectedMlSummary?.heavyRain != null
                      ? (selectedMlSummary.heavyRain * 100).toFixed(1)
                      : forecast.forecast.heavy_rain_probability
                  }
                  icon="⛈"
                />

                <div className="metric anomaly">
                  <div className="micon">Δ</div>

                  <div>
                    <div className="mtitle">{t(lang, 'anomaly')}</div>

                    <div className="value">
                      {selectedMlSummary?.anomalyMm != null
                        ? `${selectedMlSummary.anomalyMm > 0 ? '+' : ''}${selectedMlSummary.anomalyMm.toFixed(1)} mm`
                        : `${forecast.forecast.rainfall_anomaly > 0 ? '+' : ''}${forecast.forecast.rainfall_anomaly}%`}
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* REAL SPATIAL MAP */}
            <section className="split panel">
              <div className="map-wrap">
                <div className="section-title">{t(lang, 'riskMap')}</div>

                <div className="map-head">
                  <div>
                    <b>
                      {mapHeading(lang, mapLayer)} • {horizon}{' '}
                      {lang === 'hi' ? 'दिन' : 'Days'}
                    </b>
                    <div className="spatial-map-date">
                      {t(lang, 'snapshotDate')}: {spatialMapPredictionDate || SPATIAL_PREDICTION_DATE}
                    </div>
                  </div>

                  <div
                    className="probability-legend"
                    aria-label="Farmer-friendly risk probability legend"
                    style={{ minWidth: '290px' }}
                  >
                    <div
                      style={{
                        fontSize: '12px',
                        fontWeight: 800,
                        marginBottom: '7px',
                        textAlign: 'right',
                      }}
                    >
                      {lang === 'hi'
                        ? 'रंग = जोखिम की संभावना'
                        : 'Colour = chance of the risk'}
                    </div>

                    <div
                      style={{
                        display: 'flex',
                        gap: '7px',
                        flexWrap: 'wrap',
                        justifyContent: 'flex-end',
                      }}
                    >
                      {[
                        ['low', RISK_MAP_COLORS.low, '0–34.9%', 'कम'],
                        ['moderate', RISK_MAP_COLORS.moderate, '35–54.9%', 'मध्यम'],
                        ['high', RISK_MAP_COLORS.high, '55–74.9%', 'उच्च'],
                        ['very-high', RISK_MAP_COLORS['very-high'], '75–100%', 'बहुत उच्च'],
                      ].map(([key, color, rangeEn, rangeHi]) => (
                        <span
                          key={key}
                          style={{
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '5px',
                            fontSize: '11px',
                            fontWeight: 750,
                            whiteSpace: 'nowrap',
                          }}
                        >
                          <span
                            aria-hidden="true"
                            style={{
                              width: '13px',
                              height: '13px',
                              borderRadius: '3px',
                              background: color,
                              border: '1px solid rgba(15,23,42,0.18)',
                            }}
                          />
                          {lang === 'hi' ? rangeHi : rangeEn}
                        </span>
                      ))}
                    </div>

                    <div
                      style={{
                        marginTop: '6px',
                        fontSize: '11px',
                        color: '#64748B',
                        textAlign: 'right',
                      }}
                    >
                      {lang === 'hi'
                        ? 'पंचायत पर click/tap करें = exact % और risk level'
                        : 'Click/tap a Panchayat = exact % and risk level'}
                    </div>
                  </div>
                </div>

                <div className="spatial-map-controls">
                  <label>
                    {t(lang, 'mapScope')}
                    <select
                      value={mapScope}
                      onChange={(e) => {
                        setMapScope(e.target.value);
                        setSelectedArea(null);
                      }}
                    >
                      <option value="selected">{t(lang, 'selectedPanchayat')}</option>
                      <option value="all">{t(lang, 'allPanchayats')}</option>
                    </select>
                  </label>

                  <label>
                    {t(lang, 'mapIndicator')}
                    <select
                      value={mapLayer}
                      onChange={(e) => {
                        setMapLayer(e.target.value);
                        setSelectedArea(null);
                      }}
                    >
                      {Object.entries(MAP_LAYER_DEFS).map(([id, layer]) => (
                        <option key={id} value={id}>
                          {lang === 'hi' ? layer.hi : layer.en}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>

                {(!spatialMap || spatialMapPredictionLoading) && !spatialMapError && (
                  <div className="spatial-map-state">
                    {t(lang, 'mapLoading')}
                  </div>
                )}

                {spatialMapError && (
                  <div className="spatial-map-state error">
                    {t(lang, 'mapError')} {spatialMapError}
                  </div>
                )}

                {spatialMap && spatialMapProbabilities && !spatialMapPredictionLoading && (
                  <svg
                    viewBox={`0 0 ${mapProjector?.width || 900} ${mapProjector?.height || 560}`}
                    className="risk-map-real"
                    role="img"
                    aria-label="Real Panchayat spatial probability map"
                  >
                    {renderedMapFeatures.map((feature) => (
                      <path
                        key={feature.id}
                        d={feature.path}
                        className={`map-feature ${
                          selectedArea?.id === feature.id ? 'selected-map-feature' : ''
                        }`}
                        style={{ fill: probabilityToColor(feature.probability * 100) }}
                        fillRule="evenodd"
                        clipRule="evenodd"
                        onClick={() => selectMapFeature(feature)}
                        aria-label={`${feature.label}: ${formatProbability(
                          feature.probability * 100
                        )}`}
                      >
                        <title>
                          {feature.label} •{' '}
                          {formatProbability(feature.probability * 100)}
                        </title>
                      </path>
                    ))}

                    {mapScope === 'all' && blockBoundaryData.map((boundary) => (
                      <path
                        key={`block-boundary-${boundary.blockName}`}
                        d={boundary.path}
                        fill="none"
                        stroke={boundary.color}
                        strokeWidth="2.4"
                        strokeDasharray={boundary.dash}
                        strokeLinejoin="round"
                        strokeLinecap="round"
                        vectorEffect="non-scaling-stroke"
                        pointerEvents="none"
                        aria-hidden="true"
                      />
                    ))}

                    {mapScope === 'all' && blockBoundaryAudit.valid && (
                      <g pointerEvents="none" aria-hidden="true">
                        <rect
                          x="24"
                          y="24"
                          width="188"
                          height="132"
                          rx="10"
                          fill="rgba(255,255,255,0.94)"
                          stroke="#CBD5E1"
                        />
                        <text
                          x="38"
                          y="45"
                          fontSize="13"
                          fontWeight="700"
                          fill="#0F172A"
                        >
                          Block boundaries
                        </text>
                        {Object.entries(BLOCK_BOUNDARY_DEFS).map(([blockName, definition], index) => {
                          const y = 68 + index * 22;
                          return (
                            <g key={`block-legend-${blockName}`}>
                              <line
                                x1="40"
                                x2="58"
                                y1={y}
                                y2={y}
                                stroke={definition.color}
                                strokeWidth="3"
                                strokeDasharray={definition.dash}
                                strokeLinecap="round"
/>
                              <text
                                x="68"
                                y={y + 4}
                                fontSize="12"
                                fill="#334155"
                              >
                                {blockName}
                              </text>
                            </g>
                          );
                        })}
                      </g>
                    )}
                  </svg>
                )}

                {mapScope === 'all' && !blockBoundaryAudit.valid && (
                  <div className="spatial-map-state error">
                    Block-boundary audit failed. Expected current LGD Panchayat counts do not
                    match the 486-feature spatial map: {' '}
                    {blockBoundaryAudit.details
                      .map((item) => `${item.blockName} ${item.expected}/${item.actual}`)
                      .join(' • ')}
                  </div>
                )}

                <div className="map-note">
                  {t(lang, 'exampleMap')}
                  <br />
                  {t(lang, 'mapHindcastNote')}
                </div>
              </div>

              {/* SELECTED AREA */}
              <div className="side-card">
                <div className="section-title">{t(lang, 'selectedArea')}</div>

                <div className="selected-box">
                  <div className="selected-name">
                    {selectedArea?.label ||
                      `${forecast.location.block} • ${forecast.location.panchayat}`}
                  </div>

                  <div className="probability-badge">
                    {t(lang, 'selectedProbability')}: {formatProbability(
                      getSelectedRiskValue()
                    )}
                    {getSelectedRiskValue() != null && (
                      <> · {localizedRisk(lang, getSelectedRiskValue())}</>
                    )}
                  </div>
                </div>

                <div className="mini-stats">
                  <div>
                    <span>{t(lang, 'onsetSmall')}</span>
                    <b>{onsetDisplayValue(selectedMlSummary, forecast, spatialMapPredictionDate)}</b>
                  </div>

                  <div>
                    <span>{t(lang, 'breakSmall')}</span>
                    <b>
                      {selectedMlSummary?.breakProxy != null
                        ? `${(selectedMlSummary.breakProxy * 100).toFixed(1)}%`
                        : `${forecast.forecast.break_probability}%`}
                    </b>
                  </div>

                  <div>
                    <span>{t(lang, 'heavySmall')}</span>
                    <b>
                      {selectedMlSummary?.heavyRain != null
                        ? `${(selectedMlSummary.heavyRain * 100).toFixed(1)}%`
                        : `${forecast.forecast.heavy_rain_probability}%`}
                    </b>
                  </div>

                  <div>
                    <span>{t(lang, 'horizonSmall')}</span>
                    <b>
                      {horizon} {t(lang, 'days')}
                    </b>
                  </div>
                </div>
              </div>
            </section>

            {/* ADVISORY
                Keep probability numbers in the Probabilistic Outlook.
                This card focuses on interpretation and farmer actions. */}
            <section className="panel advisory-panel">
              <div className="dash-head">
                <div>
                  <div className="section-title">{t(lang, 'advisory')}</div>

                  <h2>{t(lang, 'advisoryTitle')}</h2>
                </div>

                <div className="language-badge">{LANGUAGES.find((item) => item.code === lang)?.label}</div>
              </div>

              <div className="advisory-controls">
                <select
                  value={crop}
                  onChange={(e) => handleCropChange(e.target.value)}
                >
                  <option value="rice">{lang === 'hi' ? 'धान' : 'Rice'}</option>
                  <option value="maize">{lang === 'hi' ? 'मक्का' : 'Maize'}</option>
                </select>

                <button
                  className="secondary"
                  onClick={() => loadAdvisory(lang, crop)}
                >
                  {t(lang, 'refreshAdvisory')}
                </button>
              </div>

              {advisory && (
                <div className="advisory-card">
                  <div>
                    <div className="advisory-title">
                      {advisory.title}
                    </div>

                    <p>{getAdvisoryIntro()}</p>
                  </div>

                  <div className="actions">
                    {advisory.actions.map((action, index) => (
                      <div key={index}>
                        ✓ {action}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </section>

            {/* DELIVERY */}
            <section className="panel delivery">
              <div className="section-title">{t(lang, 'delivery')}</div>

              <div className="delivery-grid">
                <button
                  onClick={() => handleDelivery('SMS')}
                >
                  <span>📱 SMS</span>
                  <small>{t(lang, 'smsText')}</small>
                </button>

                <button
                  onClick={() => handleDelivery('WhatsApp')}
                >
                  <span>💬 WhatsApp</span>
                  <small>{t(lang, 'whatsappText')}</small>
                </button>

                <button onClick={handleVoiceCall}>
                  <span>📞 Voice Call</span>
                  <small>{t(lang, 'voiceText')}</small>
                </button>

                <div className="delivery-note">
                  <b>{t(lang, 'manyChannels')}</b>
                  <br />
                  {t(lang, 'manyChannelsText')}
                </div>
              </div>

              {deliveryStatus && (
                <div className="delivery-status">
                  <div>
                    <strong>
                      ✓ {deliveryStatus}
                    </strong>

                    {deliveryChannel && (
                      <span>
                        {' '}
                        • Channel: {deliveryChannel}
                      </span>
                    )}
                  </div>

                  <div className="delivery-message">
                    <div className="delivery-message-title">{t(lang, 'messagePreview')}</div>

                    <p>{buildDeliveryMessage()}</p>
                  </div>
                </div>
              )}
            </section>

            <section className="panel sos-panel">
              <div>
                <div className="section-title">{t(lang, 'sos')}</div>
                <h2>{t(lang, 'sosTitle')}</h2>
                <p className="sos-description">{t(lang, 'sosText')}</p>
              </div>

              <div className="sos-actions">
                <button className="sos-primary" onClick={handleSos}>
                  🚨 {t(lang, 'sendSos')}
                </button>
                <button className="sos-secondary" onClick={handleSupportCall}>
                  📞 {t(lang, 'supportCall')}
                </button>
              </div>

              {sosStatus && (
                <div className="sos-status">
                  <strong>✓ {sosStatus}</strong>
                </div>
              )}

              <div className="sos-note">{t(lang, 'sosNote')}</div>
            </section>
          </>
        )}

        <footer>
          <span>मौसमसाथी • SIH 26086</span>

          <span>{t(lang, 'demoDisclaimer')}</span>
        </footer>
      </main>

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

function Metric({ title, value, icon }) {
  return (
    <div className="metric">
      <div className="micon">{icon}</div>

      <div>
        <div className="mtitle">{title}</div>
        <div className="value">{value}%</div>
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);

// step22_map_clarity_blank_area_and_indicator_labels_installed

// step22b_footer_current_snapshot_wording_installed

// step23_farmer_risk_map_legend_neutral_boundaries_installed_v2

// step24_map_indicator_fill_block_identity_raw_calibrated_explained_v3

// step24_duplicate_block_dash_prop_fixed

// step25_farmer_first_map_simplification_calibrated_only_installed
