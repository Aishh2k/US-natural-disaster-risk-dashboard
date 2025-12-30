// Global variables
let map;
let geojsonLayer;
let currentView = 'historical'; // 'historical', 'projections', or 'risk_composite'
let currentYear = 2024;
let currentNOAAMetric = 'loss'; // 'loss' or 'fatalities'
let dashboardData = {};
let nriData = {};
let statesGeoJSON;
let selectedEvents = new Set();
let selectedState = null;
let nriChartInstances = {};

// Consistent colors for event types (donut chart)
// --- FEMA / Census style regions for coloring NRI charts ---
const REGION_COLORS = {
    "Northeast": "#4e79a7",
    "Southeast": "#f28e2b",
    "Midwest": "#59a14f",
    "West": "#e15759",
    "Other": "#bab0ab"
};

const REGION_BY_STATE = {
    // Northeast
    "Maine": "Northeast", "New Hampshire": "Northeast", "Vermont": "Northeast",
    "Massachusetts": "Northeast", "Rhode Island": "Northeast", "Connecticut": "Northeast",
    "New York": "Northeast", "New Jersey": "Northeast", "Pennsylvania": "Northeast",

    // Midwest
    "Ohio": "Midwest", "Michigan": "Midwest", "Indiana": "Midwest", "Illinois": "Midwest",
    "Wisconsin": "Midwest", "Minnesota": "Midwest", "Iowa": "Midwest", "Missouri": "Midwest",
    "North Dakota": "Midwest", "South Dakota": "Midwest", "Nebraska": "Midwest", "Kansas": "Midwest",

    // Southeast
    "Delaware": "Southeast", "Maryland": "Southeast", "District of Columbia": "Northeast",
    "Virginia": "Southeast", "West Virginia": "Southeast", "Kentucky": "Southeast",
    "North Carolina": "Southeast", "South Carolina": "Southeast",
    "Georgia": "Southeast", "Florida": "Southeast",
    "Alabama": "Southeast", "Mississippi": "Southeast", "Tennessee": "Southeast",
    "Arkansas": "Southeast", "Louisiana": "Southeast",

    // West
    "Texas": "West", "Oklahoma": "West", "Montana": "West", "Wyoming": "West",
    "Colorado": "West", "New Mexico": "West", "Arizona": "West", "Utah": "West",
    "Idaho": "West", "Nevada": "West", "California": "West", "Oregon": "West",
    "Washington": "West", "Alaska": "West", "Hawaii": "West"
};

function getRegionForState(stateName) {
    if (!stateName) return "Other";

    const s = stateName.trim().toLowerCase();

    // Handle DC in all common forms
    if (s === 'district of columbia' ||
        s === 'washington, d.c.' ||
        s === 'washington dc' ||
        s === 'dc') {
        return "Northeast";
    }

    // Try exact match with normalized keys
    for (const key in REGION_BY_STATE) {
        if (key.toLowerCase() === s) {
            return REGION_BY_STATE[key];
        }
    }

    return "Other";
}


function getRegionColor(stateName) {
    const region = getRegionForState(stateName);
    return REGION_COLORS[region] || REGION_COLORS["Other"];
}


const EVENT_COLORS = {
    // Coastal / water-related
    "Astronomical Low Tide": "#1f77b4",
    "Coastal Flood": "#2ca9df",
    "Lakeshore Flood": "#2070b4",
    "Flood": "#4fa3d1",
    "Flash Flood": "#ff7f0e",
    "High Surf": "#2a9df4",
    "Rip Current": "#1f9e89",
    "Storm Surge/Tide": "#0b559f",
    "Seiche": "#6baed6",
    "Sneakerwave": "#3182bd",
    "Tsunami": "#08519c",

    // Winter / cold
    "Blizzard": "#9ecae1",
    "Cold/Wind Chill": "#c7e9f1",
    "Extreme Cold/Wind Chill": "#6baed6",
    "Freezing Fog": "#a6bddb",
    "Frost/Freeze": "#74c476",
    "Ice Storm": "#9edae5",
    "Lake-Effect Snow": "#deebf7",
    "Heavy Snow": "#c6dbef",
    "Sleet": "#b3cde3",
    "Winter Storm": "#8c96c6",
    "Winter Weather": "#bfd3e6",

    // Fog / smoke
    "Dense Fog": "#969696",
    "Dense Smoke": "#636363",
    "Marine Dense Fog": "#bdbdbd",

    // Drought / heat / fire
    "Drought": "#bcbd22",
    "Excessive Heat": "#ff7f0e",
    "Heat": "#ffa64d",
    "Wildfire": "#d62728",

    // Wind / storms
    "Dust Devil": "#d9a66b",
    "Dust Storm": "#c49c94",
    "High Wind": "#969696",
    "Strong Wind": "#c7e9c0",
    "Thunderstorm Wind": "#31a354",
    "Marine High Wind": "#a1d99b",
    "Marine Strong Wind": "#74c476",
    "Marine Thunderstorm Wind": "#238b45",
    "Waterspout": "#41ab5d",

    // Convective / vortices
    "Funnel Cloud": "#f7b6d2",
    "Tornado": "#e377c2",

    // Tropical / hurricane
    "Hurricane": "#9467bd",
    "Hurricane (Typhoon)": "#7b52ab",
    "Marine Hurricane/Typhoon": "#6a51a3",
    "Tropical Depression": "#9e9ac8",
    "Tropical Storm": "#bc80bd",
    "Marine Tropical Depression": "#807dba",
    "Marine Tropical Storm": "#8c6bb1",

    // Other precip
    "Heavy Rain": "#66c2a5",
    "Hail": "#8c564b",
    "Marine Hail": "#c7a76b",

    // Cryo / slope
    "Avalanche": "#1f78b4",
    "Debris Flow": "#8c6d31",

    // Lightning etc.
    "Lightning": "#ffd92f",
    "Marine Lightning": "#e5c100",

    // Volcanic / geomag
    "Volcanic Ash": "#b15928",
    "Volcanic Ashfall": "#8c2d04",
    "Northern Lights": "#6a3d9a",

    // Currents & misc marine
    "Rip Current": "#1f9e89",

    // safety fallback
    "default": "#999999"
};

const HAZARD_SUGGESTIONS = {
    // Flooding & coastal
    "Flood": [
        "For Flood:",
        "Please avoid walking or driving through standing water.",
        "It may be safest to move to higher ground during flood alerts.",
        "Keeping important items in a dry, easy-to-reach place can be helpful."
    ],
    "Flash Flood": [
        "For Flash Flood:",
        "Moving to higher ground quickly can greatly increase safety.",
        "Please try not to cross fast-moving water on foot or in a vehicle.",
        "Staying aware of alerts can help you react early."
    ],
    "Coastal Flood": [
        "For Coastal Flood:",
        "Staying away from the shoreline during coastal flooding can reduce risk.",
        "Moving valuables to higher floors may help protect them.",
        "Following local guidance about evacuation is often the safest choice."
    ],
    "Lakeshore Flood": [
        "For Lakeshore Flood:",
        "Avoid low-lying areas near the shoreline during high water.",
        "Please keep a safe distance from waves and unstable banks.",
        "Monitoring local updates can help you plan ahead."
    ],
    "Storm Surge/Tide": [
        "For Storm Surge:",
        "It may be safest to move away from low-lying coastal areas before storms arrive.",
        "Avoid walking or driving through water pushed inland by storm surge.",
        "Following evacuation instructions early can help avoid dangerous conditions."
    ],

    // Tornadoes
    "Tornado": [
        "For Tornado:",
        "If possible, shelter in an interior room or basement away from windows.",
        "Covering your head with pillows, blankets, or a helmet can offer protection.",
        "Waiting for the official all-clear helps ensure safe conditions."
    ],
    "Funnel Cloud": [
        "For Funnel Cloud:",
        "Staying indoors and following local alerts is recommended.",
        "Be ready to move to an interior room if a warning is issued.",
        "Keeping a small emergency kit nearby can be reassuring."
    ],

    // Tropical systems / hurricanes
    "Hurricane": [
        "For Hurricane:",
        "Preparing water, food, and medications ahead of time can be very helpful.",
        "Bringing loose outdoor items indoors reduces the chance of flying debris.",
        "Following evacuation guidance from officials is often the safest choice."
    ],
    "Hurricane (Typhoon)": [
        "For Typhoon:",
        "Preparing essential supplies ahead of time can make conditions easier to manage.",
        "Securing outdoor belongings can help reduce damage from strong winds.",
        "Evacuating when advised helps keep you and your household safe."
    ],
    "Tropical Storm": [
        "For Tropical Storm:",
        "Heavy rain and wind may make travel difficult; planning ahead is helpful.",
        "Bringing loose items indoors can prevent them from being blown around.",
        "Staying updated on local advisories can guide safe decisions."
    ],
    "Tropical Depression": [
        "For Tropical Depression:",
        "Even weaker systems can bring heavy rain; caution while traveling is helpful.",
        "Keeping drains and gutters clear can reduce localized flooding.",
        "Checking forecasts regularly helps you prepare for changing conditions."
    ],

    // Wildfire
    "Wildfire": [
        "For Wildfire:",
        "Keeping windows and doors closed can help reduce smoke indoors.",
        "Having a small emergency bag ready can make evacuation less stressful.",
        "If evacuation is advised, leaving calmly and early can help avoid traffic and hazards."
    ],
    "Dense Smoke": [
        "For Dense Smoke:",
        "Staying indoors with windows closed can reduce smoke exposure.",
        "Using masks designed for air quality (when available) may be helpful outdoors.",
        "Checking on neighbors who may be sensitive to smoke is a kind gesture."
    ],

    // Winter weather
    "Winter Storm": [
        "For Winter Storm:",
        "Limiting travel during heavy snow or ice can reduce accidents.",
        "Keeping extra blankets and warm clothing nearby can be reassuring.",
        "Staying updated on forecasts may help you plan errands around the storm."
    ],
    "Winter Weather": [
        "For Winter Weather:",
        "Taking extra care when walking or driving on slippery surfaces is important.",
        "Wearing warm layers can help protect against cold conditions.",
        "Checking in on neighbors or family members can be very thoughtful."
    ],
    "Ice Storm": [
        "For Ice Storm:",
        "Staying indoors during icy conditions is often the safest choice.",
        "Avoiding standing under trees or power lines can reduce risk from falling branches.",
        "Having flashlights and extra batteries available is helpful during outages."
    ],
    "Blizzard": [
        "For Blizzard:",
        "If possible, stay indoors while visibility is very low.",
        "Avoid traveling during whiteout conditions when you cannot see the road clearly.",
        "Keeping an emergency kit in your home and car can be especially useful."
    ],
    "Heavy Snow": [
        "For Heavy Snow:",
        "Clearing snow carefully and taking breaks can prevent overexertion.",
        "Using proper footwear can help avoid slips and falls.",
        "Allowing extra travel time can make winter driving safer."
    ],

    // Heat / cold / drought
    "Excessive Heat": [
        "For Excessive Heat:",
        "Drinking water regularly can help your body stay cool.",
        "Spending time in shaded or air-conditioned spaces can reduce heat stress.",
        "Checking on older adults, children, and pets is especially helpful."
    ],
    "Heat": [
        "For Heat:",
        "Trying to avoid strenuous activity during the hottest part of the day can be helpful.",
        "Light clothing and frequent water breaks may make you more comfortable.",
        "Using fans or visiting cooler public spaces can reduce discomfort."
    ],
    "Drought": [
        "For Drought:",
        "Using water thoughtfully can support your community during dry periods.",
        "Following local water-use guidelines is greatly appreciated.",
        "Keeping plants and lawns on reduced watering schedules can make a difference."
    ],
    "Cold/Wind Chill": [
        "For Cold/Wind Chill",
        "Wearing warm layers and covering exposed skin can help prevent frostbite.",
        "Limiting time outdoors during very low wind-chill values is recommended.",
        "Bringing pets indoors can help keep them safe and comfortable."
    ],
    "Extreme Cold/Wind Chill": [
        "For Extreme Cold/Wind Chill:",
        "Spending as little time as possible outside can reduce health risks.",
        "Ensuring heaters and generators are used safely can prevent accidents.",
        "Checking on neighbors or relatives can be especially supportive."
    ],

    // Wind / thunderstorms
    "Thunderstorm Wind": [
        "For Thunderstorm Wind:",
        "Securing outdoor items can help prevent them from being blown away.",
        "Staying away from windows during strong wind is often safest.",
        "Parking away from trees or power lines can reduce damage to vehicles."
    ],
    "Strong Wind": [
        "For Strong Wind:",
        "Remaining indoors during very strong wind can help keep you safe.",
        "Avoiding areas with loose debris or unstable structures is important.",
        "Following local advisories can help you know when conditions improve."
    ],
    "High Wind": [
        "For High Wind:",
        "Closing windows and doors firmly can reduce drafts and noise.",
        "Staying clear of tall trees and power lines can lower the risk of injury.",
        "Planning travel around the most intense wind periods may be helpful."
    ],

    // Lightning
    "Lightning": [
        "For Lightning:",
        "Moving indoors when you hear thunder is usually the safest choice.",
        "Avoiding showers, sinks, and electronics during storms can reduce risk.",
        "Waiting about 30 minutes after the last thunder before going back outside is a good rule."
    ],
    "Marine Lightning": [
        "Marine Lightning:",
        "It is often safest to come ashore or seek shelter when storms approach.",
        "Avoiding open water during lightning is strongly recommended.",
        "Checking marine forecasts before going out can be very helpful."
    ],

    // Hail
    "Hail": [
        "For Hail:",
        "Staying inside during hail can help avoid injury.",
        "Parking vehicles under cover, if possible, may reduce damage.",
        "Closing curtains or blinds can prevent broken glass from entering rooms."
    ],
    "Marine Hail": [
        "For Marine Hail:",
        "Seeking safe harbor when hail is in the forecast is recommended.",
        "Wearing protective gear on deck can reduce injury from falling hail.",
        "Keeping updated on marine weather statements is very helpful."
    ],

    // Heavy rain
    "Heavy Rain": [
        "For Heavy Rain:",
        "Driving slowly and leaving extra space between vehicles can increase safety.",
        "Avoiding areas known to flood easily is a good precaution.",
        "Clearing drains and gutters can help water flow away from your home."
    ],

    // Coastal / special
    "Rip Current": [
        "For Rip Current",
        "Swimming near lifeguards is often the safest option.",
        "If caught in a rip current, trying to swim parallel to the shore can help.",
        "Checking beach flags or signs before entering the water is very helpful."
    ],
    "High Surf": [
        "For High Surf:",
        "Staying off rocks and jetties during high surf can reduce fall risk.",
        "Watching waves from a safe distance is recommended.",
        "Following beach advisories helps keep everyone safe."
    ],
    "Tsunami": [
        "For Tsunami:",
        "Moving inland or to higher ground when a tsunami warning is issued is very important.",
        "Avoiding coastal areas until officials declare them safe is strongly advised.",
        "Following evacuation routes and signage can guide you to safer locations."
    ],

    // Fallback
    "_default": [
        "General guidance:",
        "Staying aware of local weather alerts can be very helpful.",
        "Preparing a small emergency kit can make unexpected events easier to manage.",
        "Following guidance from local officials is often the safest choice."
    ]
};

// ===========================
// INIT
// ===========================
document.addEventListener('DOMContentLoaded', async () => {
    map = L.map('map').setView([25.8283, -96.5795], 3.5);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    await loadData();
    initializeSlider();
    initializeDropdown();
    initializeNOAAMetricToggle();
    initializeChatbot();
    updateView();

    // Add legend
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function () {
        const div = L.DomUtil.create('div', 'info legend');
        div.innerHTML = getLegendContent();
        return div;
    };

    legend.addTo(map);
    const safetyListElem = document.getElementById('safety-list');
    if (safetyListElem) {
        safetyListElem.innerHTML = '';
        const tips = HAZARD_SUGGESTIONS["_default"];
        tips.forEach(msg => {
            const li = document.createElement('li');
            li.textContent = msg;
            safetyListElem.appendChild(li);
        });
    }
});


async function loadData() {
    try {
        const [dataResponse, nriResponse, geojsonResponse, predictionsResponse] = await Promise.all([
            fetch('datasets/noaa_data.json'),
            fetch('datasets/nri_data.json'),
            fetch('datasets/us-states.json'),
            fetch('datasets/predictions_data.json')
        ]);
        dashboardData = await dataResponse.json();
        nriData = await nriResponse.json();
        statesGeoJSON = await geojsonResponse.json();

        // Load predictions data
        const predictionsData = await predictionsResponse.json();
        dashboardData.projections = predictionsData;

        console.log("Data loaded successfully");

        if (dashboardData.unique_event_types) {
            dashboardData.unique_event_types.forEach(type => selectedEvents.add(type));
        } else {
            let eventTypes = new Set();
            if (dashboardData.historical) {
                for (const year in dashboardData.historical) {
                    const yearData = dashboardData.historical[year];
                    for (const state in yearData) {
                        const stateData = yearData[state];
                        if (stateData.events && Array.isArray(stateData.events)) {
                            stateData.events.forEach(event => {
                                eventTypes.add(event.type);
                            });
                        }
                    }
                }
            }
            eventTypes.forEach(type => selectedEvents.add(type));
        }
    } catch (error) {
        console.error("Failed to load data:", error);
    }
}


function initializeSlider() {
    const slider = document.getElementById('year-slider');
    const yearDisplay = document.getElementById('year-display');

    slider.addEventListener('input', (e) => {
        currentYear = parseInt(e.target.value);
        yearDisplay.textContent = currentYear;
        if (currentView === 'historical') {
            updateMapLayer();
            updateCharts();
        }
    });
    yearDisplay.textContent = currentYear;
}

function initializeDropdown() {
    const dropdown = document.getElementById('view-metric');
    dropdown.addEventListener('change', (e) => {
        const val = e.target.value;

        if (val === 'historical') {
            currentView = 'historical';
        } else if (val === 'projections') {
            currentView = 'projections';
        } else if (val === 'risk_composite') {
            currentView = 'risk_composite';
        }

        updateView();
    });

    dropdown.value = 'historical';
    currentView = 'historical';
}


function initializeNOAAMetricToggle() {
    const toggleGroup = document.getElementById('noaa-metric-toggle');
    if (!toggleGroup) return;

    const radios = toggleGroup.querySelectorAll('input[name="noaa-metric"]');
    radios.forEach(r => {
        r.addEventListener('change', (e) => {
            currentNOAAMetric = e.target.value; // 'loss' or 'fatalities'
            if (currentView === 'historical') {
                updateMapLayer();
                updateCharts();
                updateLegend();
            }
        });
    });
}

// ===========================
// VIEW SWITCH
// ===========================
function updateView() {
    const yearControl = document.getElementById('year-control');
    const filtersPanel = document.getElementById('filters-panel');
    const analyticsRow = document.getElementById('analytics-row');
    const nriAnalyticsRow = document.getElementById('nri-analytics-row');
    const predictionPanel = document.getElementById('prediction-panel');
    const nriRadarPanel = document.getElementById('nri-radar-panel');
    const noaaToggle = document.getElementById('noaa-metric-toggle');
    const chatbotBtn = document.getElementById('chatbot-toggle');

    if (currentView === 'historical') {
        yearControl.style.display = 'block';
        filtersPanel.style.display = 'block';
        analyticsRow.style.display = 'flex';
        nriAnalyticsRow.style.display = 'none';

        predictionPanel.style.display = 'block';
        nriRadarPanel.style.display = 'none';

        if (noaaToggle) noaaToggle.style.display = 'flex';
        if (chatbotBtn) chatbotBtn.style.display = 'none';

        showHistoricalView();
    } else if (currentView === 'projections') {
        yearControl.style.display = 'none';
        filtersPanel.style.display = 'none';
        analyticsRow.style.display = 'flex';
        nriAnalyticsRow.style.display = 'none';

        predictionPanel.style.display = 'none';
        nriRadarPanel.style.display = 'none';

        if (noaaToggle) noaaToggle.style.display = 'none';
        if (chatbotBtn) chatbotBtn.style.display = 'flex';

        showProjectionsView();
    } else {
        yearControl.style.display = 'none';
        filtersPanel.style.display = 'none';
        analyticsRow.style.display = 'none';
        nriAnalyticsRow.style.display = 'flex';

        predictionPanel.style.display = 'none';
        nriRadarPanel.style.display = 'block';

        if (noaaToggle) noaaToggle.style.display = 'none';
        if (chatbotBtn) chatbotBtn.style.display = 'none';

        showRiskView();
    }
    updateLegend();
}

function showHistoricalView() {
    initializeEventFilters();
    updateMapLayer();
    updateCharts();
}

function showProjectionsView() {
    currentYear = 2025;
    updateMapLayer();
    updateCharts();
}

function showRiskView() {
    updateMapLayer();
    updateNRICharts(null); // Show national average initially
}

// ===========================
// EVENT FILTERS
// ===========================
function initializeEventFilters() {
    let eventTypes = new Set();

    if (dashboardData.unique_event_types) {
        eventTypes = new Set(dashboardData.unique_event_types);
    } else if (dashboardData.historical) {
        for (const year in dashboardData.historical) {
            const yearData = dashboardData.historical[year];
            for (const state in yearData) {
                const stateData = yearData[state];
                if (stateData.events && Array.isArray(stateData.events)) {
                    stateData.events.forEach(event => {
                        eventTypes.add(event.type);
                    });
                }
            }
        }
    }

    const container = document.getElementById('event-filter-dropdown');
    container.className = 'multiselect-dropdown';
    container.innerHTML = '';

    const btn = document.createElement('div');
    btn.className = 'multiselect-btn';
    btn.textContent = 'Select Events (All Selected)';

    const content = document.createElement('div');
    content.className = 'multiselect-content';

    btn.addEventListener('click', (e) => {
        e.stopPropagation();
        content.classList.toggle('show');
    });

    document.addEventListener('click', (e) => {
        if (!container.contains(e.target)) {
            content.classList.remove('show');
        }
    });

    const sortedTypes = Array.from(eventTypes).sort();
    const checkboxes = [];

    // Select All Logic
    const selectAllCb = document.getElementById('select-all-events');

    function updateSelectAllState() {
        const allChecked = checkboxes.every(cb => cb.checked);
        selectAllCb.checked = allChecked;

        const checkedCount = checkboxes.filter(cb => cb.checked).length;
        if (checkedCount === 0) {
            btn.textContent = 'Select specific events...';
        } else if (checkedCount === sortedTypes.length) {
            btn.textContent = 'Select Events (All Selected)';
        } else {
            btn.textContent = `Select Events (${checkedCount} Selected)`;
        }
    }

    selectAllCb.addEventListener('change', (e) => {
        const isChecked = e.target.checked;
        checkboxes.forEach(cb => {
            cb.checked = isChecked;
            if (isChecked) {
                selectedEvents.add(cb.value);
            } else {
                selectedEvents.delete(cb.value);
            }
        });
        updateSelectAllState();
        if (currentView === 'historical') {
            updateMapLayer();
            updateCharts();
        }
    });

    sortedTypes.forEach(type => {
        const item = document.createElement('label');
        item.className = 'multiselect-item';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.value = type;
        checkbox.checked = true;
        checkboxes.push(checkbox);

        checkbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                selectedEvents.add(type);
            } else {
                selectedEvents.delete(type);
            }

            updateSelectAllState();

            if (currentView === 'historical') {
                updateMapLayer();
                updateCharts();
            }
        });

        item.appendChild(checkbox);
        item.appendChild(document.createTextNode(type));
        content.appendChild(item);
    });

    container.appendChild(btn);
    container.appendChild(content);
}

// ===========================
// MAP LAYER UPDATE
// ===========================
function updateMapLayer() {
    if (geojsonLayer) {
        map.removeLayer(geojsonLayer);
    }

    geojsonLayer = L.geoJSON(statesGeoJSON, {
        style: feature => {
            const stateName = feature.properties.name;
            const value = getStateValue(stateName);
            const isSelected = stateName === selectedState;
            return {
                fillColor: getColor(value),
                weight: isSelected ? 3 : 1,
                opacity: 1,
                color: isSelected ? '#fff' : '#666',
                fillOpacity: isSelected ? 0.9 : 0.7
            };
        },
        onEachFeature: (feature, layer) => {
            layer.on({
                mouseover: e => highlightFeature(e, feature),
                mouseout: resetHighlight,
                click: e => selectState(e, feature)
            });
        }
    }).addTo(map);

    updateLegend();
}

function getStateValue(stateName) {
    if (currentView === 'historical') {
        if (!dashboardData.historical || !dashboardData.historical[currentYear]) {
            return 0;
        }

        const yearData = dashboardData.historical[currentYear];
        const stateData = yearData[stateName];
        if (!stateData) return 0;

        let totalLoss = 0;
        let totalFatalities = 0;

        if (stateData.events && Array.isArray(stateData.events) && selectedEvents.size > 0) {
            for (const event of stateData.events) {
                if (selectedEvents.has(event.type)) {
                    totalLoss += event.loss || 0;
                    totalFatalities += event.fatalities || 0;
                }
            }
        } else {
            totalLoss = stateData.loss || 0;
            totalFatalities = stateData.fatalities || 0;
        }

        if (currentNOAAMetric === 'fatalities') {
            return totalFatalities;
        } else {
            return totalLoss;
        }
    } else if (currentView === 'projections') {
        if (!dashboardData.projections || !dashboardData.projections['2025']) {
            return 0;
        }

        const projData = dashboardData.projections['2025'];
        const stateData = projData[stateName];
        if (!stateData) return 0;

        return stateData.loss || 0;
    } else if (currentView === 'risk_composite') {
        const stateNRI = nriData[stateName];
        if (!stateNRI) return 0;

        const riskScore = stateNRI.risk_score || 0;
        if (riskScore < 0) return 0;
        return stateNRI.eal_total || 0;
    }

    return 0;
}

function getColor(value) {
    // =============================
    // PROJECTIONS (2025 Predictions)
    // =============================
    if (currentView === 'projections') {
        // Scale optimized for actual data range: ~$10k to ~$50M (most states $300k-$3M)
        if (value > 10000000) return '#67000d';   // > $10M (darkest red)
        if (value > 5000000) return '#a50f15';    // $5M - $10M
        if (value > 2000000) return '#cb181d';    // $2M - $5M
        if (value > 1000000) return '#ef3b2c';    // $1M - $2M (median area)
        if (value > 500000) return '#fb6a4a';     // $500k - $1M
        if (value > 200000) return '#fc9272';     // $200k - $500k
        if (value > 100000) return '#fcbba1';     // $100k - $200k
        if (value > 0) return '#fee5d9';          // $0 - $100k (lightest)
        return '#555';
    }

    // =============================
    // HISTORICAL (Economic Loss)
    // =============================
    if (currentView === 'historical' && currentNOAAMetric === 'loss') {
        const millions = value / 1e6;
        if (millions > 500) return '#8B0000'; // > 500M
        if (millions > 100) return '#B22222'; // 100M - 500M
        if (millions > 50) return '#DC143C'; // 50M - 100M
        if (millions > 10) return '#FF6347'; // 10M - 50M
        if (millions > 0) return '#FFA07A'; // 0 - 10M
        return '#555';
    }

    // =============================
    // HISTORICAL (Fatalities)
    // =============================
    if (currentView === 'historical' && currentNOAAMetric === 'fatalities') {
        if (value > 100) return '#8B0000';
        if (value > 50) return '#B22222';
        if (value > 10) return '#DC143C';
        if (value > 5) return '#FF6347';
        if (value > 0) return '#FFA07A';
        return '#555';
    }

    // =============================
    // NRI Composite (Billions)
    // =============================
    const billions = value / 1e9;
    if (billions > 2) return '#8B0000';
    if (billions > 1) return '#B22222';
    if (billions > 0.5) return '#DC143C';
    if (billions > 0.2) return '#FF6347';
    if (billions > 0) return '#FFA07A';
    return '#555';
}

// ===========================
// MAP INTERACTIONS
// ===========================
function highlightFeature(e, feature) {
    const layer = e.target;
    layer.setStyle({
        weight: 2.3,
        color: '#ffffffd6',
        fillOpacity: 0.8
    });
    layer.bringToFront();

    updateSidebar(feature.properties.name);

    if (currentView === 'historical') {
        updateCharts(feature.properties.name);
    } else if (currentView === 'projections') {
        updateProjectionsCharts(feature.properties.name);
    } else if (currentView === 'risk_composite') {
        updateNRICharts(feature.properties.name);
    }
}

function resetHighlight(e) {
    geojsonLayer.resetStyle(e.target);

    if (currentView === 'historical') {
        updateCharts(null); // Reset to National
    } else if (currentView === 'projections') {
        updateProjectionsCharts(null);
    } else if (currentView === 'risk_composite') {
        updateNRICharts(null);
    }

    // Always show generic tips when not hovering a specific state
    const safetyListElem = document.getElementById('safety-list');
    if (safetyListElem) {
        safetyListElem.innerHTML = '';
        const tips = HAZARD_SUGGESTIONS["_default"];
        tips.forEach(msg => {
            const li = document.createElement('li');
            li.textContent = msg;
            safetyListElem.appendChild(li);
        });
    }
}

function selectState(e, feature) {
    const stateName = feature.properties.name;

    // Toggle selection
    if (selectedState === stateName) {
        selectedState = null; // Deselect
    } else {
        selectedState = stateName; // Select new state
    }

    // Refresh map to update styling
    updateMapLayer();

    // Update visualizations based on selected state
    if (selectedState) {
        updateSidebar(selectedState);
        if (currentView === 'historical') {
            updateCharts(selectedState);
        } else if (currentView === 'projections') {
            updateProjectionsCharts(selectedState);
        } else if (currentView === 'risk_composite') {
            updateNRICharts(selectedState);
        }
    } else {
        // Reset to national view
        if (currentView === 'historical') {
            updateCharts(null);
        } else if (currentView === 'projections') {
            updateProjectionsCharts(null);
        } else if (currentView === 'risk_composite') {
            updateNRICharts(null);
        }

        const safetyListElem = document.getElementById('safety-list');
        if (safetyListElem) {
            safetyListElem.innerHTML = '';
            const tips = HAZARD_SUGGESTIONS["_default"];
            tips.forEach(msg => {
                const li = document.createElement('li');
                li.textContent = msg;
                safetyListElem.appendChild(li);
            });
        }
    }
}


function updateSidebar(stateName) {
    const detailsContent = document.getElementById('details-content');
    const instruction = document.querySelector('#state-info .instruction');

    instruction.style.display = 'none';
    detailsContent.style.display = 'block';

    document.getElementById('state-name').textContent = stateName;

    if (currentView === 'historical') {
        updateHistoricalDetails(stateName);
    } else if (currentView === 'projections') {
        updateProjectionsDetails(stateName);
    } else if (currentView === 'risk_composite') {
        updateRiskDetails(stateName);
    }
}

function updateHistoricalDetails(stateName) {
    document.getElementById('noaa-details').style.display = 'block';
    document.getElementById('fatalities-row').style.display = 'block';
    document.getElementById('risk-metrics').style.display = 'none';

    if (!dashboardData.historical || !dashboardData.historical[currentYear]) {
        document.getElementById('state-loss').textContent = '$0';
        document.getElementById('state-fatalities').textContent = '0';
        document.getElementById('state-events').innerHTML = '<li>No data for this year</li>';
        return;
    }

    const yearData = dashboardData.historical[currentYear];
    const stateData = yearData[stateName];

    if (!stateData) {
        document.getElementById('state-loss').textContent = '$0';
        document.getElementById('state-fatalities').textContent = '0';
        document.getElementById('state-events').innerHTML = '<li>No data for this year</li>';
        return;
    }

    let totalLoss = 0;
    let totalFatalities = 0;
    const eventSummary = {};

    if (stateData.events && Array.isArray(stateData.events)) {
        for (const event of stateData.events) {
            if (selectedEvents.has(event.type)) {
                totalLoss += event.loss || 0;
                totalFatalities += event.fatalities || 0;

                if (!eventSummary[event.type]) {
                    eventSummary[event.type] = { count: 0, loss: 0 };
                }
                eventSummary[event.type].count += 1;
                eventSummary[event.type].loss += event.loss || 0;
            }
        }
    }

    document.getElementById('state-loss').textContent = formatCurrency(totalLoss);
    document.getElementById('state-fatalities').textContent = totalFatalities.toLocaleString();

    const eventList = Object.entries(eventSummary)
        .map(([type, data]) => ({ type, ...data }))
        .sort((a, b) => b.loss - a.loss)
        .slice(0, 5);

    const eventsList = document.getElementById('state-events');
    eventsList.innerHTML = '';

    if (eventList.length === 0) {
        eventsList.innerHTML = '<li>No events match current filters</li>';
    } else {
        eventList.forEach(evt => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${evt.type}</strong>: ${evt.count} events, ${formatCurrency(evt.loss)}`;
            eventsList.appendChild(li);
        });
    }

    updatePrediction(stateName);

    // Safety tips based on top hazard
    const safetyListElem = document.getElementById('safety-list');
    if (safetyListElem) {
        safetyListElem.innerHTML = '';

        if (eventList.length > 0) {
            const topHazard = eventList[0].type;
            const tips = HAZARD_SUGGESTIONS[topHazard] || HAZARD_SUGGESTIONS["_default"];

            tips.forEach(msg => {
                const li = document.createElement('li');
                li.textContent = msg;
                safetyListElem.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = "Safety tips will appear here when hazard information is available.";
            safetyListElem.appendChild(li);
        }
    }
}

function updateRiskDetails(stateName) {
    document.getElementById('noaa-details').style.display = 'none';
    document.getElementById('fatalities-row').style.display = 'none';
    document.getElementById('risk-metrics').style.display = 'block';

    const stateNRI = nriData[stateName];
    if (stateNRI) {
        document.getElementById('risk-rating').textContent =
            stateNRI.risk_score ? stateNRI.risk_score.toFixed(2) : 'N/A';
        document.getElementById('sovi-score').textContent =
            stateNRI.sovi_score ? stateNRI.sovi_score.toFixed(2) : 'N/A';
        document.getElementById('resl-score').textContent =
            stateNRI.resl_score ? stateNRI.resl_score.toFixed(2) : 'N/A';
        document.getElementById('state-loss').textContent =
            formatCurrency(stateNRI.eal_total || 0);

        updateRadarChart(stateNRI);
    }

    // Safety tips in FEMA mode: use general guidance (or later we can
    // customize based on hazard types if you add them to NRI data).
    const safetyListElem = document.getElementById('safety-list');
    if (safetyListElem) {
        safetyListElem.innerHTML = '';
        const tips = HAZARD_SUGGESTIONS["_default"];
        tips.forEach(msg => {
            const li = document.createElement('li');
            li.textContent = msg;
            safetyListElem.appendChild(li);
        });
    }
}


function updatePrediction(stateName) {
    const lossElem = document.getElementById('pred-loss');
    const fatalitiesElem = document.getElementById('pred-fatalities');

    if (!dashboardData.historical) {
        lossElem.textContent = 'N/A';
        fatalitiesElem.textContent = 'N/A';
        return;
    }

    const years = [];
    const losses = [];
    const fatalities = [];

    for (let year = 2000; year <= 2024; year++) {
        if (dashboardData.historical[year]) {
            let yearLoss = 0;
            let yearFatalities = 0;

            if (stateName) {
                const stateData = dashboardData.historical[year][stateName];
                if (stateData && stateData.events) {
                    stateData.events.forEach(event => {
                        if (selectedEvents.has(event.type)) {
                            yearLoss += event.loss || 0;
                            yearFatalities += event.fatalities || 0;
                        }
                    });
                }
            } else {
                // National
                for (const state in dashboardData.historical[year]) {
                    const stateData = dashboardData.historical[year][state];
                    if (stateData.events) {
                        stateData.events.forEach(event => {
                            if (selectedEvents.has(event.type)) {
                                yearLoss += event.loss || 0;
                                yearFatalities += event.fatalities || 0;
                            }
                        });
                    }
                }
            }

            years.push(year);
            losses.push(yearLoss);
            fatalities.push(yearFatalities);
        }
    }

    if (years.length < 5) {
        lossElem.textContent = 'Insufficient Data';
        fatalitiesElem.textContent = 'Insufficient Data';
        return;
    }

    const predLoss = calculateRegression(years, losses, 2025);
    const predFatalities = calculateRegression(years, fatalities, 2025);

    lossElem.textContent = formatCurrency(Math.max(0, predLoss));
    fatalitiesElem.textContent = Math.max(0, Math.round(predFatalities)).toLocaleString();
}

function calculateRegression(x, y, predictX) {
    const n = x.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumXX = 0;
    for (let i = 0; i < n; i++) {
        sumX += x[i];
        sumY += y[i];
        sumXY += x[i] * y[i];
        sumXX += x[i] * x[i];
    }
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    return slope * predictX + intercept;
}

// ===========================
// NOAA CHARTS (still LOSS-based)
// ===========================
let monthlyChart, eventDistChart, benchmarkChart;

function updateCharts(stateName = null) {
    if (currentView === 'projections') {
        updateProjectionsCharts(stateName);
        return;
    }
    updateMonthlyChart(stateName);
    updateEventDistChart(stateName);
    updateBenchmarkChart(stateName);
}

function updateMonthlyChart(stateName) {
    const ctx = document.getElementById('monthly-chart');
    const monthlyData = new Array(12).fill(0);

    const metricIsFatalities = (currentNOAAMetric === 'fatalities');
    const metricLabel = metricIsFatalities ? 'Fatalities' : 'Loss (Millions USD)';
    const seriesLabel = metricIsFatalities ? 'Total Fatalities' : 'Total Loss (Millions)';

    const title = document.querySelector('#monthly-chart').parentElement.previousElementSibling;
    title.textContent = stateName
        ? `Monthly ${metricLabel} Trend (${stateName})`
        : `Monthly ${metricLabel} Trend (National)`;

    if (dashboardData.historical && dashboardData.historical[currentYear]) {
        const yearData = dashboardData.historical[currentYear];
        const processState = (sData) => {
            if (sData.events && Array.isArray(sData.events)) {
                sData.events.forEach(event => {
                    if (selectedEvents.has(event.type)) {
                        const month = (event.month || 1) - 1;
                        if (month >= 0 && month < 12) {
                            if (metricIsFatalities) {
                                monthlyData[month] += event.fatalities || 0;
                            } else {
                                monthlyData[month] += (event.loss || 0) / 1e6; // Millions
                            }
                        }
                    }
                });
            }
        };

        if (stateName) {
            if (yearData[stateName]) processState(yearData[stateName]);
        } else {
            for (const state in yearData) {
                processState(yearData[state]);
            }
        }
    }

    const labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    if (monthlyChart) monthlyChart.destroy();
    monthlyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: seriesLabel,
                data: monthlyData,
                borderColor: 'rgba(255, 194, 10, 1)',
                backgroundColor: 'rgba(255, 194, 10, 0.25)',
                fill: true,
                tension: 0.1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: metricLabel }
                }
            }
        }
    });
}


function updateEventDistChart(stateName) {
    const ctx = document.getElementById('event-dist-chart');
    const eventTotals = {};

    const metricIsFatalities = (currentNOAAMetric === 'fatalities');
    const seriesLabel = metricIsFatalities ? 'Total Fatalities' : 'Total Loss (Millions)';
    const metricLabelShort = metricIsFatalities ? 'Fatalities' : 'Loss';
    const title = document.querySelector('#event-dist-chart').parentElement.previousElementSibling;

    title.textContent = stateName
        ? `Event Type Distribution (${metricLabelShort} – ${stateName})`
        : `Event Type Distribution (${metricLabelShort} – National)`;

    if (dashboardData.historical && dashboardData.historical[currentYear]) {
        const yearData = dashboardData.historical[currentYear];
        const processState = (sData) => {
            if (sData.events && Array.isArray(sData.events)) {
                sData.events.forEach(event => {
                    if (selectedEvents.has(event.type)) {
                        if (!eventTotals[event.type]) eventTotals[event.type] = 0;
                        if (metricIsFatalities) {
                            eventTotals[event.type] += event.fatalities || 0;
                        } else {
                            eventTotals[event.type] += event.loss || 0;
                        }
                    }
                });
            }
        };

        if (stateName) {
            if (yearData[stateName]) processState(yearData[stateName]);
        } else {
            for (const state in yearData) {
                processState(yearData[state]);
            }
        }
    }

    const sortedEvents = Object.entries(eventTotals)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 8);

    const labels = sortedEvents.map(([name]) => name);
    const data = sortedEvents.map(([, value]) =>
        metricIsFatalities ? value : value / 1e6 // millions
    );

    if (eventDistChart) eventDistChart.destroy();

    eventDistChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                label: seriesLabel,
                data: data,
                backgroundColor: labels.map(name => EVENT_COLORS[name] || EVENT_COLORS.default),
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '40%',
            plugins: {
                legend: {
                    display: true,
                    position: 'right',
                    labels: {
                        boxWidth: 10,
                        font: { size: 10 }
                    }
                }
            }
        }
    });
}


function updateBenchmarkChart(stateName) {
    const ctx = document.getElementById('benchmark-chart');
    const yearlyTotals = {};
    const metricIsFatalities = (currentNOAAMetric === 'fatalities');

    const metricLabel = metricIsFatalities ? 'Fatalities' : 'Loss (Millions USD)';
    const seriesLabel = metricIsFatalities ? 'Total Fatalities' : 'Total Loss (Millions)';

    const title = document.querySelector('#benchmark-chart').parentElement.previousElementSibling;
    title.textContent = stateName
        ? `Historical Context (${metricLabel} – ${stateName})`
        : `Historical Context (${metricLabel} – National)`;

    for (let year = 2000; year <= 2024; year++) {
        yearlyTotals[year] = 0;
    }

    if (dashboardData.historical) {
        for (const year in dashboardData.historical) {
            const y = parseInt(year);
            const yearData = dashboardData.historical[year];

            const processState = (sData) => {
                if (sData.events && Array.isArray(sData.events)) {
                    sData.events.forEach(event => {
                        if (selectedEvents.has(event.type)) {
                            if (metricIsFatalities) {
                                yearlyTotals[y] += event.fatalities || 0;
                            } else {
                                yearlyTotals[y] += (event.loss || 0) / 1e6; // Millions
                            }
                        }
                    });
                }
            };

            if (stateName) {
                if (yearData[stateName]) processState(yearData[stateName]);
            } else {
                for (const state in yearData) {
                    processState(yearData[state]);
                }
            }
        }
    }

    const labels = Object.keys(yearlyTotals);
    const data = Object.values(yearlyTotals);
    const backgroundColors = labels.map(year =>
        parseInt(year) === currentYear
            ? 'rgba(0, 90, 181, 1)'
            : 'rgba(170, 210, 255, 0.9)'
    );

    if (benchmarkChart) benchmarkChart.destroy();
    benchmarkChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: seriesLabel,
                data: data,
                backgroundColor: backgroundColors
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    title: { display: true, text: metricLabel }
                }
            }
        }
    });
}


// ===========================
// NRI CHARTS
// ===========================
function updateNRICharts(stateName) {
    const nationalAverages = calculateNationalAverages();
    updateRiskQuadrant(stateName, nationalAverages);
    updateEquityGap(stateName, nationalAverages);
    updateRiskEfficiency(stateName, nationalAverages);
}

function calculateNationalAverages() {
    let sumRisk = 0, sumSovi = 0, sumResl = 0, sumEal = 0, count = 0;
    let sumGap = 0; // Risk - Resilience

    for (const state in nriData) {
        const d = nriData[state];
        if (d && d.risk_score >= 0 && d.sovi_score >= 0 && d.resl_score >= 0) {
            sumRisk += d.risk_score || 0;
            sumSovi += d.sovi_score || 0;
            sumResl += d.resl_score || 0;
            sumEal += d.eal_total || 0;
            sumGap += (d.risk_score || 0) - (d.resl_score || 0);
            count++;
        }
    }
    return {
        avgRisk: count ? sumRisk / count : 0,
        avgSovi: count ? sumSovi / count : 0,
        avgResl: count ? sumResl / count : 0,
        avgEal: count ? sumEal / count : 0,
        avgGap: count ? sumGap / count : 0
    };
}

/* ---------------------------
   Risk Quadrant (SoVI vs Risk)
   --------------------------- */
function updateNRICharts(stateName) {
    const nationalAverages = calculateNationalAverages();
    updateRiskQuadrant(stateName, nationalAverages);
    updateEquityGap(stateName, nationalAverages);
    updateRiskEfficiency(stateName, nationalAverages);
}

function updateRiskQuadrant(stateName, nationalAverages) {
    const ctx = document.getElementById('risk-quadrant-chart');
    const datasets = [];

    // Group states by region
    const regionGroups = {
        "Northeast": [],
        "Southeast": [],
        "Midwest": [],
        "West": [],
        "Other": []
    };

    for (const state in nriData) {
        const d = nriData[state];
        if (d && d.risk_score >= 0 && d.sovi_score >= 0) {
            const region = getRegionForState(state);
            regionGroups[region].push({
                x: d.sovi_score,
                y: d.risk_score,
                state
            });
        }
    }

    // One dataset per region so legend shows NE/SE/MW/W/Other
    Object.keys(regionGroups).forEach(region => {
        if (regionGroups[region].length === 0) return;
        datasets.push({
            label: region,
            data: regionGroups[region],
            backgroundColor: REGION_COLORS[region],
            pointRadius: 6,
            pointHoverRadius: 7
        });
    });

    // Highlight selected state on top (white dot with dark outline)
    if (stateName && nriData[stateName] &&
        nriData[stateName].risk_score >= 0 &&
        nriData[stateName].sovi_score >= 0) {

        datasets.push({
            label: stateName,
            data: [{
                x: nriData[stateName].sovi_score,
                y: nriData[stateName].risk_score,
                state: stateName
            }],
            backgroundColor: '#ffffff',
            borderColor: '#000000',
            borderWidth: 1.5,
            pointRadius: 9,
            pointHoverRadius: 10
        });
    }

    const avgSovi = nationalAverages.avgSovi;
    const avgRisk = nationalAverages.avgRisk;

    // Quadrant lines + text labels
    const quadrantGuides = {
        id: 'quadrantGuides',
        afterDraw(chart) {
            const { ctx, chartArea: { left, right, top, bottom } } = chart;
            const xScale = chart.scales.x;
            const yScale = chart.scales.y;
            if (!xScale || !yScale) return;

            const xMid = xScale.getPixelForValue(avgSovi);
            const yMid = yScale.getPixelForValue(avgRisk);

            ctx.save();

            // dashed lines
            ctx.strokeStyle = '#777';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);

            // vertical (SoVI)
            ctx.beginPath();
            ctx.moveTo(xMid, bottom);
            ctx.lineTo(xMid, top);
            ctx.stroke();

            // horizontal (Risk)
            ctx.beginPath();
            ctx.moveTo(left, yMid);
            ctx.lineTo(right, yMid);
            ctx.stroke();

            ctx.setLineDash([]);

            // quadrant labels
            ctx.fillStyle = '#bbbbbb';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';

            // Top-left: High Risk / Lower SoVI
            ctx.fillText('High Risk / Lower SoVI',
                (left + xMid) / 2, top + 12);

            // Top-right: High Risk / High SoVI
            ctx.fillText('High Risk / High SoVI',
                (xMid + right) / 2, top + 12);

            // Bottom-left: Lower Risk / Lower SoVI
            ctx.fillText('Lower Risk / Lower SoVI',
                (left + xMid) / 2, bottom - 12);

            // Bottom-right: Lower Risk / High SoVI
            ctx.fillText('Lower Risk / High SoVI',
                (xMid + right) / 2, bottom - 12);

            ctx.restore();
        }
    };

    if (nriChartInstances.riskQuadrant) {
        nriChartInstances.riskQuadrant.destroy();
    }

    nriChartInstances.riskQuadrant = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Social Vulnerability (SoVI Score)' },
                    min: 0,
                    max: 100
                },
                y: {
                    title: { display: true, text: 'Risk Score' },
                    min: 0,
                    max: 100
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Risk Quadrant (SoVI vs Risk Score)'
                },
                legend: {
                    display: true,
                    labels: { usePointStyle: true }
                },
                tooltip: {
                    callbacks: {
                        label(context) {
                            const p = context.raw;
                            const d = nriData[p.state] || {};
                            const risk = (d.risk_score ?? p.y).toFixed(1);
                            const sovi = (d.sovi_score ?? p.x).toFixed(1);
                            const resl = (d.resl_score ?? 0).toFixed(1);
                            return `${p.state}: Risk ${risk}, SoVI ${sovi}, Resilience ${resl}`;
                        }
                    }
                }
            }
        },
        plugins: [quadrantGuides]
    });
}

/* ---------------------------
   Social Damage Susceptibility (Top 10 + Selected)
   Index = Risk Score × SoVI Score
   --------------------------- */
function updateEquityGap(stateName, nationalAverages) {
    const ctx = document.getElementById('equity-gap-chart');

    // 1. Build list with Risk, SoVI, and composite index
    const validStates = Object.entries(nriData)
        .filter(([name, d]) => d && d.risk_score >= 0 && d.sovi_score >= 0)
        .map(([name, d]) => {
            const risk = d.risk_score || 0;
            const sovi = d.sovi_score || 0;
            const index = risk * sovi;  // Social Damage Susceptibility Index
            return { name, risk, sovi, index };
        });

    if (!validStates.length) return;

    // 2. Sort by susceptibility index (descending)
    validStates.sort((a, b) => b.index - a.index);

    // 3. Top 10 by index
    const top10 = validStates.slice(0, 10);

    // 4. Check selected state rank in full list
    let selectedInfo = null;
    if (stateName) {
        const selIdx = validStates.findIndex(s => s.name === stateName);
        if (selIdx !== -1) {
            const rank = selIdx + 1; // 1-based
            selectedInfo = { ...validStates[selIdx], rank };
        }
    }

    // 5. Build final list to plot
    const plotStates = top10.map((s, i) => ({
        ...s,
        rank: i + 1
    }));

    if (selectedInfo) {
        const inTop10 = plotStates.find(s => s.name === selectedInfo.name);
        if (!inTop10 && selectedInfo.rank > 10) {
            // add selected as extra bar with its true rank
            plotStates.push({
                ...selectedInfo,
                rank: selectedInfo.rank
            });
        }
    }

    const labels = plotStates.map(s => `#${s.rank} ${s.name}`);
    const data = plotStates.map(s => s.index);


    const baseColor = '#C1666B';
    const highlightColor = '#FCE8EF';


    const backgroundColors = plotStates.map(s =>
        (selectedInfo && s.name === selectedInfo.name) ? highlightColor : baseColor
    );

    if (nriChartInstances.equityGap) {
        nriChartInstances.equityGap.destroy();
    }

    nriChartInstances.equityGap = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: backgroundColors
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    beginAtZero: true,
                    min: 0,
                    max: 4400, // full range 0–4400
                    title: {
                        display: true,
                        text: 'Social Damage Susceptibility Index (Risk × SoVI)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Top States Socially Vulnerable to Natural Hazards'
                    },
                    ticks: {
                        autoSkip: false   // show all ranks 1–10
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Social Damage Susceptibility (Top 10 States)'
                },
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label(context) {
                            const s = plotStates[context.dataIndex];
                            const risk = s.risk.toFixed(1);
                            const sovi = s.sovi.toFixed(1);
                            const indexVal = s.index.toFixed(1);
                            return [
                                `${s.name} (Rank #${s.rank})`,
                                `Risk: ${risk}`,
                                `SoVI: ${sovi}`,
                                `Index: ${indexVal}`
                            ];
                        }
                    }
                }
            }
        }
    });
}


/* ---------------------------
   Risk Efficiency (EAL vs Resilience)
   Color by region, bubble size = SoVI
   --------------------------- */
function updateRiskEfficiency(stateName, nationalAverages) {
    const ctx = document.getElementById('risk-efficiency-chart');
    const datasets = [];

    const regionGroups = {
        "Northeast": [],
        "Southeast": [],
        "Midwest": [],
        "West": [],
        "Other": []
    };

    for (const state in nriData) {
        const d = nriData[state];
        if (d && d.eal_total >= 0 && d.resl_score >= 0) {
            const region = getRegionForState(state);
            regionGroups[region].push({
                x: d.eal_total / 1e9,  // billions
                y: d.resl_score,
                state
            });
        }
    }

    // region datasets
    Object.keys(regionGroups).forEach(region => {
        if (regionGroups[region].length === 0) return;
        datasets.push({
            label: region,
            data: regionGroups[region],
            backgroundColor: REGION_COLORS[region],
            pointRadius: 6,
            pointHoverRadius: 7
        });
    });

    // highlight selected state
    if (stateName && nriData[stateName] &&
        nriData[stateName].eal_total >= 0 &&
        nriData[stateName].resl_score >= 0) {

        datasets.push({
            label: stateName,
            data: [{
                x: nriData[stateName].eal_total / 1e9,
                y: nriData[stateName].resl_score,
                state: stateName
            }],
            backgroundColor: '#ffffff',
            borderColor: '#000000',
            borderWidth: 1.5,
            pointRadius: 9,
            pointHoverRadius: 10
        });
    }

    // Plugin: only show mean EAL & mean Resilience cross (NO regression line)
    const guidesPlugin = {
        id: 'riskEfficiencyGuides',
        afterDraw(chart) {
            const { ctx, chartArea: { left, right, top, bottom } } = chart;
            const xScale = chart.scales.x;
            const yScale = chart.scales.y;
            if (!xScale || !yScale) return;

            ctx.save();
            ctx.strokeStyle = '#777';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);

            const xAvgPix = xScale.getPixelForValue(nationalAverages.avgEal / 1e9);
            const yAvgPix = yScale.getPixelForValue(nationalAverages.avgResl);

            // vertical avg EAL
            ctx.beginPath();
            ctx.moveTo(xAvgPix, bottom);
            ctx.lineTo(xAvgPix, top);
            ctx.stroke();

            // horizontal avg Resilience
            ctx.beginPath();
            ctx.moveTo(left, yAvgPix);
            ctx.lineTo(right, yAvgPix);
            ctx.stroke();

            ctx.setLineDash([]);
            ctx.fillStyle = '#bbbbbb';
            ctx.font = '10px sans-serif';

            ctx.textAlign = 'left';
            ctx.textBaseline = 'top';
            ctx.fillText('Avg EAL', xAvgPix + 3, top + 3);

            ctx.textAlign = 'right';
            ctx.textBaseline = 'bottom';
            ctx.fillText('Avg Resilience', right - 3, yAvgPix - 3);

            ctx.restore();
        }
    };

    if (nriChartInstances.riskEfficiency) {
        nriChartInstances.riskEfficiency.destroy();
    }

    nriChartInstances.riskEfficiency = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: { display: true, text: 'Expected Annual Loss (Billions USD)' },
                    min: 0
                },
                y: {
                    title: { display: true, text: 'Resilience Score' },
                    min: 0,
                    max: 100
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Risk Efficiency (EAL vs Resilience)'
                },
                legend: {
                    display: true,
                    labels: { usePointStyle: true }
                },
                tooltip: {
                    callbacks: {
                        label(context) {
                            const p = context.raw;
                            const d = nriData[p.state] || {};
                            const eal = (d.eal_total / 1e9).toFixed(2);
                            const resl = (d.resl_score ?? p.y).toFixed(1);
                            const sovi = (d.sovi_score ?? 0).toFixed(1);
                            return `${p.state}: EAL $${eal}B, Resilience ${resl}, SoVI ${sovi}`;
                        }
                    }
                }
            }
        },
        plugins: [guidesPlugin]
    });
}


// ===========================
// RADAR CHART (NRI Risk Profile)
// ===========================
let radarChart;

function updateRadarChart(stateNRI) {
    const ctx = document.getElementById('nri-radar-chart');
    const labels = ['Risk Score', 'SoVI Score', 'Resilience Score'];
    const data = [
        stateNRI.risk_score < 0 ? 0 : stateNRI.risk_score,
        stateNRI.sovi_score < 0 ? 0 : stateNRI.sovi_score,
        stateNRI.resl_score < 0 ? 0 : stateNRI.resl_score
    ];

    if (radarChart) {
        radarChart.data.labels = labels;
        radarChart.data.datasets[0].label = stateNRI.name || 'State';
        radarChart.data.datasets[0].data = data;
        radarChart.update();
        return;
    }

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels,
            datasets: [{
                label: stateNRI.name || 'State',
                data,
                fill: true,
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgb(255, 99, 132)',
                pointBackgroundColor: 'rgb(255, 99, 132)',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgb(255, 99, 132)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    beginAtZero: true,
                    max: 100,
                    angleLines: { display: true },
                    pointLabels: { font: { size: 10 } }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (ctx) {
                            const label = ctx.label || '';
                            const v = ctx.parsed.r;
                            return `${label}: ${v.toFixed(1)}`;
                        }
                    }
                }
            }
        }
    });
}




// ===========================
// LEGEND
// ===========================
function updateLegend() {
    const legendDiv = document.querySelector('.info.legend');
    if (legendDiv) {
        legendDiv.innerHTML = getLegendContent();
    }
}

// function getLegendContent() {
//     // Historical NOAA
//     if (currentView === 'historical') {
//         // Fatalities legend
//         if (currentNOAAMetric === 'fatalities') {
//             let content = `<h4>Fatalities</h4>`;
//             const thresholds = [0, 5, 10, 50, 100];

//             for (let i = 0; i < thresholds.length; i++) {
//                 const from = thresholds[i];
//                 const to = thresholds[i + 1];
//                 const sampleValue = to ? from + 0.1 : from + 100;

//                 const color = getColor(sampleValue);
//                 content += `<i style="background:${color}"></i> `;
//                 content += to ? `${from}&ndash;${to}` : `${from}+`;
//                 content += '<br>';
//             }
//             content += '<i style="background:#555"></i> No data<br>';
//             return content;
//         }

//         // Economic loss legend (Millions)
//         let content = `<h4>Loss (Millions)</h4>`;
//         const grades = [0, 10, 50, 100, 500];

//         for (let i = 0; i < grades.length; i++) {
//             const from = grades[i];
//             const to = grades[i + 1];
//             const color = getColor((from * 1e6) + 1);
//             content += `<i style="background:${color}"></i> `;
//             content += to ? `${from}&ndash;${to}M` : `${from}+M`;
//             content += '<br>';
//         }
//         content += '<i style="background:#555"></i> No data<br>';
//         // NRI Mode
//         let content = `<h4>EAL (Billions)</h4>`;
//         const grades = [0, 0.2, 0.5, 1, 2];

//         for (let i = 0; i < grades.length; i++) {
//             const from = grades[i];
//             const to = grades[i + 1];
//             const color = getColor((from * 1e9) + 1);
//             content += `<i style="background:${color}"></i> `;
//             content += to ? `${from}&ndash;${to}B` : `${from}+B`;
//             content += '<br>';
//         }
//         content += '<i style="background:#555"></i> No data<br>';
//         return content;
//     }

//     // ===========================
//     // UTILITY
//     // ===========================
//     function formatCurrency(value) {
//         if (value >= 1e9) {
//             return `$${(value / 1e9).toFixed(2)} Billion`;
//         }
//         if (value >= 1e6) {
//             return `$${(value / 1e6).toFixed(2)} Million`;
//         }
//         if (value >= 1e3) {
//             return `$${(value / 1e3).toFixed(2)} Thousand`;
//         }
//         return `$${value.toFixed(2)}`;
//     }


function getLegendContent() {
    if (currentView === 'historical') {
        if (currentNOAAMetric === 'fatalities') {
            let content = `<h4>Fatalities</h4>`;
            const thresholds = [0, 5, 10, 50, 100];
            for (let i = 0; i < thresholds.length; i++) {
                const from = thresholds[i];
                const to = thresholds[i + 1];
                const sampleValue = to ? from + 0.1 : from + 100;
                const color = getColor(sampleValue);
                content += `<i style="background:${color}"></i> `;
                content += to ? `${from}&ndash;${to}` : `${from}+`;
                content += '<br>';
            }
            content += '<i style="background:#555"></i> No data<br>';
            return content;
        }
        let content = `<h4>Loss (Millions)</h4>`;
        const grades = [0, 10, 50, 100, 500];
        for (let i = 0; i < grades.length; i++) {
            const from = grades[i];
            const to = grades[i + 1];
            const color = getColor((from * 1e6) + 1);
            content += `<i style="background:${color}"></i> `;
            content += to ? `${from}&ndash;${to}M` : `${from}+M`;
            content += '<br>';
        }
        content += '<i style="background:#555"></i> No data<br>';
        return content;
    }
    if (currentView === 'projections') {
        let content = `<h4>Predicted Loss (2025)</h4>`;
        const grades = [0, 100, 200, 500, 1000, 2000, 5000, 10000]; // in thousands
        for (let i = 0; i < grades.length; i++) {
            const from = grades[i];
            const to = grades[i + 1];
            const color = getColor((from * 1000) + 1);
            content += `<i style="background:${color}"></i> `;
            if (from >= 1000) {
                const fromM = from / 1000;
                const toM = to ? to / 1000 : null;
                content += to ? `$${fromM}M–$${toM}M` : `$${fromM}M+`;
            } else {
                content += to ? `$${from}k–$${to}k` : `$${from}k+`;
            }
            content += '<br>';
        }
        content += '<i style="background:#555"></i> No data<br>';
        return content;
    }
    let content = `<h4>EAL (Billions)</h4>`;
    const grades = [0, 0.2, 0.5, 1, 2];
    for (let i = 0; i < grades.length; i++) {
        const from = grades[i];
        const to = grades[i + 1];
        const color = getColor((from * 1e9) + 1);
        content += `<i style="background:${color}"></i> `;
        content += to ? `${from}&ndash;${to}B` : `${from}+B`;
        content += '<br>';
    }
    content += '<i style="background:#555"></i> No data<br>';
    return content;
}

function formatCurrency(value) {
    if (value >= 1e9) {
        return `$${(value / 1e9).toFixed(2)} Billion`;
    }
    if (value >= 1e6) {
        return `$${(value / 1e6).toFixed(2)} Million`;
    }
    if (value >= 1e3) {
        return `$${(value / 1e3).toFixed(2)} Thousand`;
    }
    return `$${value.toFixed(2)}`;
}

// ===========================
// CHATBOT LOGIC
// ===========================
function initializeChatbot() {
    const chatbotToggle = document.getElementById('chatbot-toggle');
    const chatbotPanel = document.getElementById('chatbot-panel');
    const chatbotClose = document.getElementById('chatbot-close');
    const chatbotSend = document.getElementById('chatbot-send');
    const chatbotInput = document.getElementById('chatbot-input');
    const chatbotMessages = document.getElementById('chatbot-messages');

    if (!chatbotToggle || !chatbotPanel) return;

    // Toggle Chatbot
    chatbotToggle.addEventListener('click', () => {
        chatbotPanel.style.display = (chatbotPanel.style.display === 'none' || chatbotPanel.style.display === '') ? 'flex' : 'none';
        if (chatbotPanel.style.display === 'flex') {
            chatbotInput.focus();
        }
    });

    // Close Chatbot
    chatbotClose.addEventListener('click', () => {
        chatbotPanel.style.display = 'none';
    });

    // Send Message
    function sendMessage() {
        const message = chatbotInput.value.trim();
        if (!message) return;

        // Add User Message
        addMessage(message, 'user');
        chatbotInput.value = '';

        // Simulate AI Response (Mock for now)
        setTimeout(() => {
            const response = generateMockResponse(message);
            addMessage(response, 'bot');
        }, 500);
    }

    chatbotSend.addEventListener('click', sendMessage);
    chatbotInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    function addMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.classList.add('message', sender);
        msgDiv.textContent = text;
        chatbotMessages.appendChild(msgDiv);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
    }

    function generateMockResponse(query) {
        query = query.toLowerCase();
        if (query.includes('prediction') || query.includes('2025')) {
            return "Based on our models, 2025 is projected to see increased storm activity in the Southeast.";
        } else if (query.includes('california')) {
            return "California is projected to face significant wildfire risks in 2025, with estimated losses over $5 Billion.";
        } else if (query.includes('texas')) {
            return "Texas is projected to see high hail and tornado activity in 2025.";
        } else {
            return "I can help you with 2025 disaster projections. Ask me about specific states or event types!";
        }
    }
}

// ===========================
// PROJECTIONS VIEW LOGIC
// ===========================

function updateProjectionsDetails(stateName) {
    document.getElementById('noaa-details').style.display = 'block';
    document.getElementById('fatalities-row').style.display = 'block';
    document.getElementById('risk-metrics').style.display = 'none';

    if (!dashboardData.projections || !dashboardData.projections['2025']) {
        document.getElementById('state-loss').textContent = 'No Data';
        document.getElementById('state-fatalities').textContent = '0';
        document.getElementById('state-events').innerHTML = '<li>No data available</li>';
        return;
    }

    const projData = dashboardData.projections['2025'];
    const stateData = projData[stateName];

    if (!stateData) {
        document.getElementById('state-loss').textContent = '$0';
        document.getElementById('state-fatalities').textContent = '0';
        document.getElementById('state-events').innerHTML = '<li>No projected events</li>';
        return;
    }

    // Update Loss and Fatalities
    document.getElementById('state-loss').textContent = formatCurrency(stateData.loss || 0);
    document.getElementById('state-fatalities').textContent = Math.round(stateData.fatalities || 0).toLocaleString();

    // Update Top Events
    const eventsList = document.getElementById('state-events');
    eventsList.innerHTML = '';

    if (stateData.events && stateData.events.length > 0) {
        // Sort by loss descending
        const sortedEvents = [...stateData.events].sort((a, b) => (b.loss || 0) - (a.loss || 0));
        const top3 = sortedEvents.slice(0, 3);

        top3.forEach(event => {
            const li = document.createElement('li');
            li.textContent = `${event.type}: ${formatCurrency(event.loss)}`;
            eventsList.appendChild(li);
        });
    } else {
        eventsList.innerHTML = '<li>No specific events projected</li>';
    }
}

function updateProjectionsCharts(stateName) {
    updateMonthlyChartProjections(stateName);
    updateEventDistChartProjections(stateName);
    updateBenchmarkChartProjections(stateName);
}

function updateMonthlyChartProjections(stateName) {
    const ctx = document.getElementById('monthly-chart');
    const monthlyData = new Array(12).fill(0);
    const metricIsFatalities = (currentNOAAMetric === 'fatalities');
    const metricLabel = metricIsFatalities ? 'Predicted Fatalities' : 'Predicted Loss (Millions USD)';

    const title = document.querySelector('#monthly-chart').parentElement.previousElementSibling;
    title.textContent = stateName
        ? `2025 Projected Monthly Trend (${stateName})`
        : `2025 Projected Monthly Trend (National)`;

    if (dashboardData.projections && dashboardData.projections['2025']) {
        const projData = dashboardData.projections['2025'];

        const processState = (sData) => {
            if (sData.events && Array.isArray(sData.events)) {
                sData.events.forEach(event => {
                    const month = (event.month || 1) - 1;
                    if (month >= 0 && month < 12) {
                        if (metricIsFatalities) {
                            monthlyData[month] += event.fatalities || 0;
                        } else {
                            monthlyData[month] += (event.loss || 0) / 1e6; // Millions
                        }
                    }
                });
            }
        };

        if (stateName) {
            if (projData[stateName]) processState(projData[stateName]);
        } else {
            for (const state in projData) {
                processState(projData[state]);
            }
        }
    }

    const labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    if (monthlyChart) monthlyChart.destroy();
    monthlyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: metricLabel,
                data: monthlyData,
                borderColor: '#d62728', // Red for predictions
                backgroundColor: 'rgba(214, 39, 40, 0.25)',
                fill: true,
                tension: 0.1,
                borderDash: [5, 5] // Dashed line for predictions
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, title: { display: true, text: metricLabel } }
            }
        }
    });
}

function updateEventDistChartProjections(stateName) {
    const ctx = document.getElementById('event-dist-chart');
    const eventTotals = {};
    const metricIsFatalities = (currentNOAAMetric === 'fatalities');
    const metricLabelShort = metricIsFatalities ? 'Fatalities' : 'Loss';

    const title = document.querySelector('#event-dist-chart').parentElement.previousElementSibling;
    title.textContent = stateName
        ? `2025 Projected Event Distribution (${stateName})`
        : `2025 Projected Event Distribution (National)`;

    if (dashboardData.projections && dashboardData.projections['2025']) {
        const projData = dashboardData.projections['2025'];

        const processState = (sData) => {
            if (sData.events && Array.isArray(sData.events)) {
                sData.events.forEach(event => {
                    if (!eventTotals[event.type]) eventTotals[event.type] = 0;
                    if (metricIsFatalities) {
                        eventTotals[event.type] += event.fatalities || 0;
                    } else {
                        eventTotals[event.type] += event.loss || 0;
                    }
                });
            }
        };

        if (stateName) {
            if (projData[stateName]) processState(projData[stateName]);
        } else {
            for (const state in projData) {
                processState(projData[state]);
            }
        }
    }

    const sortedEvents = Object.entries(eventTotals)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 8);

    const labels = sortedEvents.map(([name]) => name);
    const data = sortedEvents.map(([, value]) => metricIsFatalities ? value : value / 1e6);

    if (eventDistChart) eventDistChart.destroy();
    eventDistChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                label: `Projected ${metricLabelShort}`,
                data: data,
                backgroundColor: labels.map(name => EVENT_COLORS[name] || EVENT_COLORS.default),
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '40%',
            plugins: {
                legend: { display: true, position: 'right', labels: { boxWidth: 10, font: { size: 10 } } }
            }
        }
    });
}

function updateBenchmarkChartProjections(stateName) {
    const ctx = document.getElementById('benchmark-chart');
    const yearlyTotals = {};
    const metricIsFatalities = (currentNOAAMetric === 'fatalities');
    const metricLabel = metricIsFatalities ? 'Fatalities' : 'Loss (Millions USD)';

    const title = document.querySelector('#benchmark-chart').parentElement.previousElementSibling;
    title.textContent = stateName
        ? `Historical + 2025 Projection (${stateName})`
        : `Historical + 2025 Projection (National)`;

    // 1. Historical Data (2000-2024)
    for (let year = 2000; year <= 2024; year++) {
        yearlyTotals[year] = 0;
        if (dashboardData.historical && dashboardData.historical[year]) {
            const yearData = dashboardData.historical[year];
            const processState = (sData) => {
                if (sData.events) {
                    sData.events.forEach(event => {
                        if (metricIsFatalities) {
                            yearlyTotals[year] += event.fatalities || 0;
                        } else {
                            yearlyTotals[year] += (event.loss || 0) / 1e6;
                        }
                    });
                }
            };
            if (stateName) {
                if (yearData[stateName]) processState(yearData[stateName]);
            } else {
                for (const state in yearData) processState(yearData[state]);
            }
        }
    }

    // 2. Projections (2025)
    yearlyTotals[2025] = 0;
    if (dashboardData.projections && dashboardData.projections['2025']) {
        const projData = dashboardData.projections['2025'];
        const processState = (sData) => {
            if (metricIsFatalities) {
                yearlyTotals[2025] += sData.fatalities || 0;
            } else {
                yearlyTotals[2025] += (sData.loss || 0) / 1e6;
            }
        };
        if (stateName) {
            if (projData[stateName]) processState(projData[stateName]);
        } else {
            for (const state in projData) processState(projData[state]);
        }
    }

    const labels = Object.keys(yearlyTotals);
    const data = Object.values(yearlyTotals);
    const backgroundColors = labels.map(year =>
        parseInt(year) === 2025 ? '#d62728' : 'rgba(170, 210, 255, 0.9)'
    );

    if (benchmarkChart) benchmarkChart.destroy();
    benchmarkChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: metricLabel,
                data: data,
                backgroundColor: backgroundColors
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, title: { display: true, text: metricLabel } }
            }
        }
    });
}
