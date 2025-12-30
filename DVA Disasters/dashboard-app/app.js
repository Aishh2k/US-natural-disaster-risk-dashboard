// Global variables
let map;
let geojsonLayer;
let currentView = 'historical'; // 'historical' or 'risk_composite'
let currentYear = 2024;
let dashboardData = {};
let nriData = {};
let statesGeoJSON;
let selectedEvents = new Set();
let nriChartInstances = {};

// Initialize the dashboard
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
    updateView();

    // Add legend
    const legend = L.control({ position: 'bottomright' });
    legend.onAdd = function (map) {
        const div = L.DomUtil.create('div', 'info legend');
        div.innerHTML = getLegendContent();
        return div;
    };
    legend.addTo(map);
});

async function loadData() {
    try {
        const timestamp = new Date().getTime();
        const [dataResponse, nriResponse, geojsonResponse] = await Promise.all([
            fetch(`dashboard_data.json?v=${timestamp}`),
            fetch(`nri_data.json?v=${timestamp}`),
            fetch('us-states.json')
        ]);
        dashboardData = await dataResponse.json();
        nriData = await nriResponse.json();
        statesGeoJSON = await geojsonResponse.json();
        console.log("Data loaded successfully");
        console.log("Dashboard structure:", Object.keys(dashboardData));
        console.log("NRI type:", typeof nriData, nriData);

        if (dashboardData.unique_event_types) {
            dashboardData.unique_event_types.forEach(type => selectedEvents.add(type));
        } else {
            // Fallback if unique_event_types is not pre-calculated
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
    const label = document.querySelector('label[for="year-slider"]');
    slider.addEventListener('input', (e) => {
        currentYear = parseInt(e.target.value);
        label.innerHTML = `Year: <br>${currentYear}`;
        if (currentView === 'historical') {
            updateMapLayer();
            updateCharts();
        }
    });
    label.innerHTML = `Year: <br>${currentYear}`;
}

function initializeDropdown() {
    const dropdown = document.getElementById('view-metric');
    dropdown.addEventListener('change', (e) => {
        currentView = e.target.value;
        updateView();
    });
}

function updateView() {
    if (currentView === 'historical') {
        document.getElementById('year-control').style.display = 'block';
        document.getElementById('filters-panel').style.display = 'block';
        document.getElementById('analytics-row').style.display = 'flex';
        document.getElementById('nri-analytics-row').style.display = 'none';

        document.getElementById('prediction-panel').style.display = 'block';
        document.getElementById('nri-radar-panel').style.display = 'none';

        // Hide Chatbot Button
        const chatbotBtn = document.getElementById('chatbot-toggle');
        if (chatbotBtn) chatbotBtn.style.display = 'none';

        showHistoricalView();
    } else if (currentView === 'projections') {
        // Projections View
        document.getElementById('year-control').style.display = 'none'; // Hide slider, fixed to 2025
        document.getElementById('filters-panel').style.display = 'none'; // No filters for now
        document.getElementById('analytics-row').style.display = 'flex'; // Reuse NOAA charts
        document.getElementById('nri-analytics-row').style.display = 'none';

        document.getElementById('prediction-panel').style.display = 'none'; // Hide prediction panel (this IS prediction)
        document.getElementById('nri-radar-panel').style.display = 'none';

        // Show Chatbot Button
        const chatbotBtn = document.getElementById('chatbot-toggle');
        if (chatbotBtn) chatbotBtn.style.display = 'flex';

        showProjectionsView();
    } else {
        document.getElementById('year-control').style.display = 'none';
        document.getElementById('filters-panel').style.display = 'none';
        document.getElementById('analytics-row').style.display = 'none';
        document.getElementById('nri-analytics-row').style.display = 'flex';

        document.getElementById('prediction-panel').style.display = 'none';
        document.getElementById('nri-radar-panel').style.display = 'block';

        // Hide Chatbot Button
        const chatbotBtn = document.getElementById('chatbot-toggle');
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

function showRiskView() {
    updateMapLayer();
    updateNRICharts(null); // Show national average initially
}

function showProjectionsView() {
    updateMapLayer();
    updateCharts();
}

// ===========================
// EVENT FILTERS
// ==========================
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
            return {
                fillColor: getColor(value),
                weight: 1,
                opacity: 1,
                color: '#666',
                fillOpacity: 0.7
            };
        },
        onEachFeature: (feature, layer) => {
            layer.on({
                mouseover: e => highlightFeature(e, feature),
                mouseout: resetHighlight,
                click: e => selectFeature(e, feature)
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

        if (!stateData) {
            return 0;
        }

        if (stateData.events && Array.isArray(stateData.events) && selectedEvents.size > 0) {
            let totalLoss = 0;
            for (const event of stateData.events) {
                if (selectedEvents.has(event.type)) {
                    totalLoss += event.loss || 0;
                }
            }
            return totalLoss;
        }

        return stateData.loss || 0;
    } else if (currentView === 'projections') {
        if (!dashboardData.projections || !dashboardData.projections['2025']) {
            return 0;
        }
        const stateData = dashboardData.projections['2025'][stateName];
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
    if (currentView === 'historical') {
        // Millions Scale for Historical Data
        const millions = value / 1e6;
        if (millions > 500) return '#8B0000'; // > 500M
        if (millions > 100) return '#B22222'; // 100M - 500M
        if (millions > 50) return '#DC143C';  // 50M - 100M
        if (millions > 10) return '#FF6347';  // 10M - 50M
        if (millions > 0) return '#FFA07A';   // 0 - 10M
        return '#555';
    } else if (currentView === 'projections') {
        // Adjusted Scale for Projections (0-50M range)
        const millions = value / 1e6;
        if (millions > 10) return '#8B0000';  // > 10M
        if (millions > 5) return '#B22222';   // 5M - 10M
        if (millions > 2) return '#DC143C';   // 2M - 5M
        if (millions > 0.5) return '#FF6347'; // 500K - 2M
        if (millions > 0) return '#FFA07A';   // 0 - 500K
        return '#555';
    } else {
        // Billions Scale (NRI)
        const billions = value / 1e9;
        if (billions > 2) return '#8B0000';   // > 2B
        if (billions > 1) return '#B22222';   // 1B - 2B
        if (billions > 0.5) return '#DC143C'; // 500M - 1B
        if (billions > 0.2) return '#FF6347'; // 200M - 500M
        if (billions > 0) return '#FFA07A';   // 0 - 200M
        return '#555';
    }
}

// ===========================
// MAP INTERACTIONS
// ===========================
// ===========================
// MAP INTERACTIONS
// ===========================
let selectedState = null;


function highlightFeature(e, feature) {
    const layer = e.target;

    // Don't highlight if another state is selected/locked
    if (selectedState && selectedState !== feature.properties.name) {
        return;
    }

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
        updateCharts(feature.properties.name);
    } else if (currentView === 'risk_composite') {
        updateNRICharts(feature.properties.name);
    }
}

function resetHighlight(e) {
    const layer = e.target;
    const feature = layer.feature;

    // If this state is selected, don't reset style
    if (selectedState === feature.properties.name) {
        return;
    }

    geojsonLayer.resetStyle(layer);

    // If a state is selected, don't reset sidebar/charts to national
    if (selectedState) {
        // Ensure the selected state is still shown (in case we hovered over it and left)
        // This might be redundant if we didn't change it, but safe
        return;
    }

    // If nothing is selected, reset to National
    if (currentView === 'historical') {
        updateCharts(null);
    } else if (currentView === 'projections') {
        updateCharts(null);
    } else if (currentView === 'risk_composite') {
        updateNRICharts(null);
    }

    // Hide sidebar details if nothing selected
    document.querySelector('#state-info .instruction').style.display = 'block';
    document.getElementById('details-content').style.display = 'none';
}


function selectFeature(e, feature) {
    const layer = e.target;
    const stateName = feature.properties.name;

    if (selectedState === stateName) {
        // Deselect
        selectedState = null;
        geojsonLayer.resetStyle(layer);

        // Reset to National
        if (currentView === 'historical') {
            updateCharts(null);
        } else if (currentView === 'projections') {
            updateCharts(null);
        } else if (currentView === 'risk_composite') {
            updateNRICharts(null);
        }

        document.querySelector('#state-info .instruction').style.display = 'block';
        document.getElementById('details-content').style.display = 'none';
    } else {
        // Select new state
        // First reset previous selection if any
        if (selectedState) {
            geojsonLayer.eachLayer(l => {
                if (l.feature.properties.name === selectedState) {
                    geojsonLayer.resetStyle(l);
                }
            });
        }

        selectedState = stateName;

        // Apply highlight style permanently
        layer.setStyle({
            weight: 2.3,
            color: '#ffffffd6',
            fillOpacity: 0.8
        });
        layer.bringToFront();

        updateSidebar(stateName);

        if (currentView === 'historical') {
            updateCharts(stateName);
        } else if (currentView === 'projections') {
            updateCharts(stateName);
        } else if (currentView === 'risk_composite') {
            updateNRICharts(stateName);
        }
    }

    L.DomEvent.stopPropagation(e); // Prevent map click
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
    updatePrediction(stateName);
}

function updateProjectionsDetails(stateName) {
    document.getElementById('noaa-details').style.display = 'block';
    document.getElementById('fatalities-row').style.display = 'block';
    document.getElementById('risk-metrics').style.display = 'none';

    if (!dashboardData.projections || !dashboardData.projections['2025']) {
        console.error("Projections data missing:", dashboardData);
        document.getElementById('state-loss').textContent = '$0';
        document.getElementById('state-fatalities').textContent = '0';
        document.getElementById('state-events').innerHTML = '<li>No data for 2025</li>';
        return;
    }

    console.log(`Updating projections for state: '${stateName}'`);
    const stateData = dashboardData.projections['2025'][stateName];
    console.log("State data found:", stateData);

    if (!stateData) {
        document.getElementById('state-loss').textContent = '$0';
        document.getElementById('state-fatalities').textContent = '0';
        document.getElementById('state-events').innerHTML = '<li>No data for 2025</li>';
        return;
    }

    document.getElementById('state-loss').textContent = formatCurrency(stateData.loss || 0);
    document.getElementById('state-fatalities').textContent = (stateData.fatalities || 0).toLocaleString();

    // For projections, we don't have detailed event list in the same way, 
    // but we constructed 'events' list in process_data.py
    const eventSummary = {};
    if (stateData.events && Array.isArray(stateData.events)) {
        stateData.events.forEach(event => {
            if (!eventSummary[event.type]) {
                eventSummary[event.type] = { count: 0, loss: 0 };
            }
            eventSummary[event.type].count += 1;
            eventSummary[event.type].loss += event.loss || 0;
        });
    }

    const eventList = Object.entries(eventSummary)
        .map(([type, data]) => ({ type, ...data }))
        .sort((a, b) => b.loss - a.loss)
        .slice(0, 5);

    const eventsList = document.getElementById('state-events');
    eventsList.innerHTML = '';

    if (eventList.length === 0) {
        eventsList.innerHTML = '<li>No events projected</li>';
    } else {
        eventList.forEach(evt => {
            const li = document.createElement('li');
            li.innerHTML = `<strong>${evt.type}</strong>: ${formatCurrency(evt.loss)}`;
            eventsList.appendChild(li);
        });
    }

    // Hide prediction panel content or show something relevant
    document.getElementById('pred-loss').textContent = 'N/A';
    document.getElementById('pred-fatalities').textContent = 'N/A';
}

function updateRiskDetails(stateName) {
    document.getElementById('noaa-details').style.display = 'none';
    document.getElementById('fatalities-row').style.display = 'none';
    document.getElementById('risk-metrics').style.display = 'block';

    const stateNRI = nriData[stateName];
    if (stateNRI) {
        document.getElementById('risk-rating').textContent = stateNRI.risk_score ? stateNRI.risk_score.toFixed(2) : 'N/A';
        document.getElementById('sovi-score').textContent = stateNRI.sovi_score ? stateNRI.sovi_score.toFixed(2) : 'N/A';
        document.getElementById('resl-score').textContent = stateNRI.resl_score ? stateNRI.resl_score.toFixed(2) : 'N/A';
        document.getElementById('state-loss').textContent = formatCurrency(stateNRI.eal_total || 0);

        updateRadarChart(stateNRI);
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
// NOAA CHARTS
// ===========================
let monthlyChart, eventDistChart, benchmarkChart;

function updateCharts(stateName = null) {
    updateMonthlyChart(stateName);
    updateEventDistChart(stateName);
    updateBenchmarkChart(stateName);
}

function updateMonthlyChart(stateName) {
    const ctx = document.getElementById('monthly-chart');
    const monthlyData = new Array(12).fill(0);
    const title = document.querySelector('#monthly-chart').parentElement.previousElementSibling;
    if (title) {
        title.textContent = stateName ? `Monthly Loss Trend (${stateName})` : `Monthly Loss Trend (National)`;
    }

    if (currentView === 'historical' && dashboardData.historical && dashboardData.historical[currentYear]) {
        const yearData = dashboardData.historical[currentYear];
        const processState = (sData) => {
            if (sData.events && Array.isArray(sData.events)) {
                sData.events.forEach(event => {
                    if (selectedEvents.has(event.type)) {
                        // event.month is 1-based integer from pandas
                        const month = (event.month || 1) - 1;
                        if (month >= 0 && month < 12) {
                            monthlyData[month] += (event.loss || 0) / 1e6; // Millions
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
    } else if (currentView === 'projections' && dashboardData.projections && dashboardData.projections['2025']) {
        const yearData = dashboardData.projections['2025'];
        const processState = (sData) => {
            if (sData.events && Array.isArray(sData.events)) {
                sData.events.forEach(event => {
                    // No filter for projections currently
                    const month = (event.month || 1) - 1;
                    if (month >= 0 && month < 12) {
                        monthlyData[month] += (event.loss || 0) / 1e6; // Millions
                    }
                });
            }
        };

        if (stateName) {
            console.log(`Filtering monthly data for state: ${stateName}`);
            if (yearData[stateName]) {
                console.log(`Found data for ${stateName}:`, yearData[stateName]);
                processState(yearData[stateName]);
            } else {
                console.log(`No data found for ${stateName}`);
            }
        } else {
            console.log('Processing all states for monthly data');
            for (const state in yearData) {
                processState(yearData[state]);
            }
        }
        console.log('Monthly chart data:', monthlyData);
    }

    const labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

    if (monthlyChart) monthlyChart.destroy();
    monthlyChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Total Loss (Millions)',
                data: monthlyData,
                borderColor: '#FF6347',
                backgroundColor: 'rgba(255, 99, 71, 0.2)',
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
                    title: { display: true, text: 'Loss (Millions USD)' }
                }
            }
        }
    });
}

function updateEventDistChart(stateName) {
    const ctx = document.getElementById('event-dist-chart');
    const eventTotals = {};
    const title = document.querySelector('#event-dist-chart').parentElement.previousElementSibling;
    title.textContent = stateName ? `Event Type Distribution (${stateName})` : `Event Type Distribution (National)`;

    if (currentView === 'historical' && dashboardData.historical && dashboardData.historical[currentYear]) {
        const yearData = dashboardData.historical[currentYear];
        const processState = (sData) => {
            if (sData.events && Array.isArray(sData.events)) {
                sData.events.forEach(event => {
                    if (selectedEvents.has(event.type)) {
                        eventTotals[event.type] = (eventTotals[event.type] || 0) + (event.loss || 0);
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
    } else if (currentView === 'projections' && dashboardData.projections && dashboardData.projections['2025']) {
        const yearData = dashboardData.projections['2025'];
        const processState = (sData) => {
            if (sData.events && Array.isArray(sData.events)) {
                sData.events.forEach(event => {
                    // No filter for projections
                    eventTotals[event.type] = (eventTotals[event.type] || 0) + (event.loss || 0);
                });
            }
        };

        if (stateName) {
            console.log(`Filtering event distribution for state: ${stateName}`);
            if (yearData[stateName]) {
                console.log(`Found data for ${stateName}`);
                processState(yearData[stateName]);
            } else {
                console.log(`No data found for ${stateName}`);
            }
        } else {
            console.log('Processing all states for event distribution');
            for (const state in yearData) {
                processState(yearData[state]);
            }
        }
        console.log('Event totals:', eventTotals);
    }

    const sortedEvents = Object.entries(eventTotals)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 8);

    const labels = sortedEvents.map(entry => entry[0]);
    const data = sortedEvents.map(entry => entry[1] / 1e6); // Millions

    if (eventDistChart) eventDistChart.destroy();

    eventDistChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                label: 'Total Loss (Millions)',
                data: data,
                backgroundColor: ['#FF6347', '#FFA07A', '#DC143C', '#B22222', '#8B0000', '#CD5C5C', '#F08080', '#FA8072'],
                hoverOffset: 4,
                borderWidth: 0
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
    const title = document.querySelector('#benchmark-chart').parentElement.previousElementSibling;
    title.textContent = stateName ? `Historical Context (${stateName})` : `Historical Context (National)`;

    if (currentView === 'projections') {
        // Only show 2025
        yearlyTotals['2025'] = 0;
        if (dashboardData.projections && dashboardData.projections['2025']) {
            const yearData = dashboardData.projections['2025'];
            const processState = (sData) => {
                yearlyTotals['2025'] += (sData.loss || 0) / 1e6;
            };

            if (stateName) {
                if (yearData[stateName]) processState(yearData[stateName]);
            } else {
                for (const state in yearData) {
                    processState(yearData[state]);
                }
            }
        }
    } else {
        // Historical 2000-2024
        for (let year = 2000; year <= 2024; year++) {
            yearlyTotals[year] = 0;
        }

        if (dashboardData.historical) {
            for (let year = 2000; year <= 2024; year++) {
                const yearStr = year.toString();
                if (dashboardData.historical[yearStr]) {
                    const yearData = dashboardData.historical[yearStr];
                    const processState = (sData) => {
                        if (sData.events && Array.isArray(sData.events)) {
                            sData.events.forEach(event => {
                                if (selectedEvents.has(event.type)) {
                                    yearlyTotals[year] += (event.loss || 0) / 1e6; // Millions
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
        }
    }

    const labels = Object.keys(yearlyTotals);
    const data = Object.values(yearlyTotals);
    const backgroundColors = labels.map(year => (parseInt(year) === currentYear || year === '2025') ? '#FF6347' : '#FFA07A');

    if (benchmarkChart) benchmarkChart.destroy();
    benchmarkChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Total Loss (Millions)',
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
                    title: { display: true, text: 'Loss (Millions USD)' }
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
    for (const state in nriData) {
        const data = nriData[state];
        if (data && data.risk_score >= 0 && data.sovi_score >= 0 && data.resl_score >= 0) {
            sumRisk += data.risk_score || 0;
            sumSovi += data.sovi_score || 0;
            sumResl += data.resl_score || 0;
            sumEal += data.eal_total || 0;
            count++;
        }
    }
    return {
        avgRisk: count > 0 ? sumRisk / count : 0,
        avgSovi: count > 0 ? sumSovi / count : 0,
        avgResl: count > 0 ? sumResl / count : 0,
        avgEal: count > 0 ? sumEal / count : 0
    };
}

function updateRiskQuadrant(stateName, nationalAverages) {
    const ctx = document.getElementById('risk-quadrant-chart');
    const datasets = [];
    const allStatesData = [];

    for (const state in nriData) {
        const data = nriData[state];
        if (data && data.risk_score >= 0 && data.sovi_score >= 0) {
            allStatesData.push({ x: data.sovi_score, y: data.risk_score, state: state });
        }
    }

    datasets.push({
        label: 'Other States',
        data: allStatesData,
        backgroundColor: 'rgba(200, 200, 200, 0.5)',
        pointRadius: 5
    });

    if (stateName && nriData[stateName] && nriData[stateName].risk_score >= 0 && nriData[stateName].sovi_score >= 0) {
        datasets.push({
            label: stateName,
            data: [{ x: nriData[stateName].sovi_score, y: nriData[stateName].risk_score }],
            backgroundColor: 'red',
            pointRadius: 8
        });
    }

    if (nriChartInstances.riskQuadrant) nriChartInstances.riskQuadrant.destroy();
    nriChartInstances.riskQuadrant = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Social Vulnerability (SoVI Score)' }, min: 0, max: 100 },
                y: { title: { display: true, text: 'Risk Score' }, min: 0, max: 100 }
            },
            plugins: {
                title: { display: true, text: 'Risk Quadrant (SoVI vs Risk Score)' },
                autocolors: false,
                annotation: {
                    annotations: {
                        lineX: { type: 'line', xMin: nationalAverages.avgSovi, xMax: nationalAverages.avgSovi, borderColor: 'black', borderWidth: 1, label: { content: 'Avg SoVI', enabled: true, position: 'start' } },
                        lineY: { type: 'line', yMin: nationalAverages.avgRisk, yMax: nationalAverages.avgRisk, borderColor: 'black', borderWidth: 1, label: { content: 'Avg Risk', enabled: true, position: 'start' } }
                    }
                }
            }
        }
    });
}

function updateEquityGap(stateName, nationalAverages) {
    const ctx = document.getElementById('equity-gap-chart');

    // 1. Get all valid states
    const validStates = Object.entries(nriData)
        .filter(([name, data]) => data.risk_score >= 0 && data.sovi_score >= 0)
        .map(([name, data]) => ({ name, ...data }));

    // 2. Sort by Risk Score Descending
    validStates.sort((a, b) => b.risk_score - a.risk_score);

    // 3. Take Top 10
    let topStates = validStates.slice(0, 10);

    // 4. Ensure selected state is included
    if (stateName && nriData[stateName]) {
        const selectedStateInTop = topStates.find(s => s.name === stateName);
        if (!selectedStateInTop) {
            const selectedStateData = validStates.find(s => s.name === stateName);
            if (selectedStateData) {
                topStates.push(selectedStateData);
                // Re-sort to keep order correct if we added it
                topStates.sort((a, b) => b.risk_score - a.risk_score);
            }
        }
    }

    // 5. Prepare Data for Chart
    const labels = topStates.map(s => s.name);
    const data = topStates.map(s => s.sovi_score);
    const backgroundColors = topStates.map(s => s.name === stateName ? 'red' : 'grey');

    if (nriChartInstances.equityGap) nriChartInstances.equityGap.destroy();
    nriChartInstances.equityGap = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Social Vulnerability Score',
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
                    max: 100,
                    title: { display: true, text: 'Social Vulnerability Score' }
                }
            },
            plugins: {
                title: { display: true, text: 'Equity Gap (Top Riskiest States)' },
                legend: { display: false }
            }
        }
    });
}

function updateRiskEfficiency(stateName, nationalAverages) {
    const ctx = document.getElementById('risk-efficiency-chart');
    const datasets = [];
    const allStatesData = [];

    for (const state in nriData) {
        const data = nriData[state];
        if (data && data.eal_total >= 0 && data.resl_score >= 0) {
            allStatesData.push({ x: data.eal_total / 1e9, y: data.resl_score, state: state }); // EAL in Billions
        }
    }

    datasets.push({
        label: 'Other States',
        data: allStatesData,
        backgroundColor: 'rgba(200, 200, 200, 0.5)',
        pointRadius: 5
    });

    if (stateName && nriData[stateName] && nriData[stateName].eal_total >= 0 && nriData[stateName].resl_score >= 0) {
        datasets.push({
            label: stateName,
            data: [{ x: nriData[stateName].eal_total / 1e9, y: nriData[stateName].resl_score }],
            backgroundColor: 'green',
            pointRadius: 8
        });
    }

    if (nriChartInstances.riskEfficiency) nriChartInstances.riskEfficiency.destroy();
    nriChartInstances.riskEfficiency = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: {
            responsive: true, maintainAspectRatio: false,
            scales: {
                x: { title: { display: true, text: 'Expected Annual Loss (Billions USD)' }, min: 0 },
                y: { title: { display: true, text: 'Resilience Score' }, min: 0, max: 100 }
            },
            plugins: {
                title: { display: true, text: 'Risk Efficiency (EAL vs Resilience)' },
                autocolors: false,
                annotation: {
                    annotations: {
                        lineX: { type: 'line', xMin: nationalAverages.avgEal / 1e9, xMax: nationalAverages.avgEal / 1e9, borderColor: 'black', borderWidth: 1, label: { content: 'Avg EAL', enabled: true, position: 'start' } },
                        lineY: { type: 'line', yMin: nationalAverages.avgResl, yMax: nationalAverages.avgResl, borderColor: 'black', borderWidth: 1, label: { content: 'Avg Resl', enabled: true, position: 'start' } }
                    }
                }
            }
        }
    });
}

// ===========================
// RADAR CHART (NRI)
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
        radarChart.data.datasets[0].data = data;
        radarChart.update();
    } else {
        radarChart = new Chart(ctx, {
            type: 'radar',
            data: {
                labels: labels,
                datasets: [{
                    label: stateNRI.name || 'State',
                    data: data,
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
                    legend: { display: false }
                }
            }
        });
    }
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

function getLegendContent() {
    let content, grades, unit, multiplier;

    if (currentView === 'historical') {
        content = `<h4>Loss (Millions)</h4>`;
        grades = [0, 10, 50, 100, 500];
        unit = 'M';
        multiplier = 1e6;
    } else if (currentView === 'projections') {
        content = `<h4>Projected Loss</h4>`;
        grades = [0, 0.5, 2, 5, 10];
        unit = 'M';
        multiplier = 1e6;
    } else {
        content = `<h4>EAL (Billions)</h4>`;
        grades = [0, 0.2, 0.5, 1, 2];
        unit = 'B';
        multiplier = 1e9;
    }

    for (let i = 0; i < grades.length; i++) {
        const from = grades[i];
        const to = grades[i + 1];
        const color = getColor(from * multiplier + 1);
        content += `<i style="background:${color}"></i> `;
        content += from + (to ? `&ndash;${to}${unit}` : `+${unit}`) + '<br>';
    }
    content += '<i style="background:#555"></i> No data<br>';
    return content;
}

// ===========================
// UTILITY
// ===========================
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
// AI CHATBOT LOGIC
// ===========================
document.addEventListener('DOMContentLoaded', () => {
    const chatbotToggle = document.getElementById('chatbot-toggle');
    const chatbotPanel = document.getElementById('chatbot-panel');
    const chatbotClose = document.getElementById('chatbot-close');
    const chatbotSend = document.getElementById('chatbot-send');
    const chatbotInput = document.getElementById('chatbot-input');
    const chatbotMessages = document.getElementById('chatbot-messages');

    if (chatbotToggle) {
        chatbotToggle.addEventListener('click', () => {
            chatbotPanel.classList.add('open');
        });
    }

    if (chatbotClose) {
        chatbotClose.addEventListener('click', () => {
            chatbotPanel.classList.remove('open');
        });
    }

    if (chatbotSend) {
        chatbotSend.addEventListener('click', sendMessage);
    }

    if (chatbotInput) {
        chatbotInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
    }

    async function sendMessage() {
        const message = chatbotInput.value.trim();
        if (!message) return;

        // Add user message
        addMessage(message, 'user');
        chatbotInput.value = '';

        // Add typing indicator
        const typingId = addTypingIndicator();

        try {
            // Prepare context (summary of 2025 projections)
            const context = getProjectionsContext();

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    context: context
                })
            });

            const data = await response.json();

            // Remove typing indicator
            removeMessage(typingId);

            if (data.error) {
                addMessage("Sorry, I encountered an error: " + data.error, 'bot');
            } else {
                addMessage(data.response, 'bot');
            }

        } catch (error) {
            removeMessage(typingId);
            addMessage("Sorry, I couldn't connect to the server. Please try again.", 'bot');
            console.error('Chatbot error:', error);
        }
    }

    function addMessage(text, sender) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `chat-message ${sender}-message`;
        msgDiv.innerHTML = `<p>${text}</p>`;
        chatbotMessages.appendChild(msgDiv);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        return msgDiv;
    }

    function addTypingIndicator() {
        const msgDiv = document.createElement('div');
        msgDiv.id = 'typing-' + Date.now();
        msgDiv.className = 'chat-message bot-message typing-indicator';
        msgDiv.innerHTML = '<p>Thinking...</p>';
        chatbotMessages.appendChild(msgDiv);
        chatbotMessages.scrollTop = chatbotMessages.scrollHeight;
        return msgDiv.id;
    }

    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function getProjectionsContext() {
        if (!dashboardData.projections || !dashboardData.projections['2025']) {
            return "No projection data available.";
        }

        // Create a summary of the data to send to AI
        const proj = dashboardData.projections['2025'];
        let summary = "2025 Disaster Projections Summary:\n";

        // Add ALL states sorted by loss
        const sortedStates = Object.entries(proj)
            .map(([name, data]) => ({ name, ...data }))
            .sort((a, b) => b.loss - a.loss);

        summary += "Projected Loss, Fatalities, and Top Risk by State:\n";
        sortedStates.forEach(s => {
            let eventInfo = "";
            if (s.events && s.events.length > 0) {
                eventInfo = `, Top Risk: ${s.events[0].type}`;
            }
            summary += `- ${s.name}: $${(s.loss / 1e6).toFixed(1)}M loss, ${s.fatalities} fatalities${eventInfo}\n`;
        });

        // Add current selected state details if any (for more detail on events)
        if (selectedState && proj[selectedState]) {
            const s = proj[selectedState];
            summary += `\nCurrently Selected State: ${selectedState}\n`;
            summary += `- Loss: $${(s.loss / 1e6).toFixed(1)}M\n`;
            summary += `- Fatalities: ${s.fatalities}\n`;
            if (s.events && s.events.length > 0) {
                summary += `- Top Events: ${s.events.slice(0, 5).map(e => e.type).join(', ')}\n`;
            }
        }

        return summary;
    }
});
