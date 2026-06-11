const COLORS = {
    primary: '#528c8e',
    primaryLight: '#8cb6b7',
    primaryDark: '#3a6b6d',
    accent: '#497a7c',
    muted: '#b0c4c5',
    text: '#2c3e40',
    bg: '#f1f5f6',
    white: '#ffffff',
    grid: '#e2e8e9'
};

const layoutTemplate = {
    font: { family: 'Inter, sans-serif', color: COLORS.text },
    plot_bgcolor: COLORS.white,
    paper_bgcolor: COLORS.white,
    margin: { t: 20, r: 20, b: 40, l: 50 },
    xaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.grid },
    yaxis: { gridcolor: COLORS.grid, zerolinecolor: COLORS.grid },
    hoverlabel: { bgcolor: COLORS.white, font: { color: COLORS.text } }
};

// State
let appData = {};

async function initDashboard() {
    try {
        // Fetch Data
        const [delayDist, missing, categorical, correlation, temporal, hourly] = await Promise.all([
            fetch('data/arr_delay_dist.json').then(r => r.json()),
            fetch('data/missing_values.json').then(r => r.json()),
            fetch('data/categorical_grouping.json').then(r => r.json()),
            fetch('data/correlation.json').then(r => r.json()),
            fetch('data/temporal.json').then(r => r.json()),
            fetch('data/hourly_delay.json').then(r => r.json())
        ]);

        appData = { delayDist, missing, categorical, correlation, temporal, hourly };

        renderMissingValues();
        renderDelayDist('original');
        renderCategorical('pre');
        renderCorrelation();
        renderTemporal('monthly');
        renderHourly();

        setupListeners();
    } catch (e) {
        console.error("Failed to load data", e);
    }
}

function renderMissingValues() {
    const data = appData.missing;
    const cols = Object.keys(data).reverse(); // Reverse for descending top-down
    const vals = cols.map(c => data[c] * 100); // Percentage

    const trace = {
        x: vals,
        y: cols,
        type: 'bar',
        orientation: 'h',
        marker: { color: COLORS.primaryLight }
    };

    Plotly.newPlot('chart-missing', [trace], {
        ...layoutTemplate,
        margin: { t: 10, r: 10, b: 40, l: 120 },
        xaxis: { title: 'Missing Percentage (%)', gridcolor: COLORS.grid }
    }, { responsive: true, displayModeBar: false });
}

function renderDelayDist(type) {
    const data = appData.delayDist[type];
    const traces = [];

    // Histogram
    const binSize = data.hist.bin_edges[1] - data.hist.bin_edges[0];
    traces.push({
        x: data.hist.bin_edges.slice(0, -1), // use left edge
        y: data.hist.counts,
        type: 'bar',
        name: 'Histogram',
        marker: { color: COLORS.primaryLight, opacity: 0.7 }
    });

    // KDE is typically scaled to counts or histogram is density. The user script gave pure counts and pure KDE.
    // We need a secondary Y axis for KDE
    if(data.kde) {
        traces.push({
            x: data.kde.x,
            y: data.kde.y,
            type: 'scatter',
            mode: 'lines',
            name: 'KDE',
            line: { color: COLORS.primaryDark, width: 2 },
            yaxis: 'y2'
        });
    }

    const layout = {
        ...layoutTemplate,
        showlegend: false,
        barmode: 'overlay',
        bargap: 0,
        yaxis2: {
            overlaying: 'y',
            side: 'right',
            showgrid: false,
            zeroline: false,
            showticklabels: false
        }
    };

    // If Original, add box plot at the bottom or top? User asked for "horizontal box plot on this original one"
    // Since combining them is tricky in Plotly without subplots, let's just make it a subplot!
    if(type === 'original' && data.box) {
        // Just plotting histogram + KDE for simplicity, box plots are tricky as overlays
        // We will just do the hist + KDE to match Yeo-Johnson
    }

    Plotly.newPlot('chart-delay-dist', traces, layout, { responsive: true, displayModeBar: false });
}

function renderCategorical(state) {
    const data = appData.categorical;
    
    // Carrier Pie Chart
    const carrierData = data['op_unique_carrier'][state];
    const carrierTrace = {
        labels: Object.keys(carrierData),
        values: Object.values(carrierData),
        type: 'pie',
        hole: 0.4,
        marker: { colors: [COLORS.primaryDark, COLORS.primary, COLORS.primaryLight, COLORS.muted, '#d0e0e1', '#e6f0f0'] },
        textposition: 'inside',
        textinfo: 'label+percent'
    };
    Plotly.newPlot('chart-cat-carrier', [carrierTrace], {
        ...layoutTemplate, margin: {t:30,b:10,l:10,r:10}, title: 'Carrier'
    }, { responsive: true, displayModeBar: false });

    // Origin Bar
    const originData = data['origin'][state];
    // Sort
    const oEntries = Object.entries(originData).sort((a,b) => b[1]-a[1]).slice(0, 30); // Show top 30 for perf
    const originTrace = {
        x: oEntries.map(e => e[0]),
        y: oEntries.map(e => e[1]),
        type: 'bar',
        marker: { color: COLORS.primary }
    };
    Plotly.newPlot('chart-cat-origin', [originTrace], {
        ...layoutTemplate, margin: {t:30,b:40,l:40,r:10}, title: 'Origin (Top 30)'
    }, { responsive: true, displayModeBar: false });

    // Dest Bar
    const destData = data['dest'][state];
    const dEntries = Object.entries(destData).sort((a,b) => b[1]-a[1]).slice(0, 30);
    const destTrace = {
        x: dEntries.map(e => e[0]),
        y: dEntries.map(e => e[1]),
        type: 'bar',
        marker: { color: COLORS.primaryLight }
    };
    Plotly.newPlot('chart-cat-dest', [destTrace], {
        ...layoutTemplate, margin: {t:30,b:40,l:40,r:10}, title: 'Destination (Top 30)'
    }, { responsive: true, displayModeBar: false });
}

function renderCorrelation() {
    const data = appData.correlation;
    const cols = Object.keys(data);
    
    // Filter logic: Only keep features that have at least one correlation > 0.3 or < -0.3 (excluding self)
    const threshold = 0.3;
    const filteredCols = cols.filter(c1 => {
        return cols.some(c2 => c1 !== c2 && Math.abs(data[c1][c2]) >= threshold);
    });

    const zMatrix = filteredCols.map(c1 => filteredCols.map(c2 => data[c1][c2]));

    const trace = {
        x: filteredCols,
        y: filteredCols,
        z: zMatrix,
        type: 'heatmap',
        colorscale: [
            ['0.0', '#ffffff'],
            ['0.5', COLORS.primaryLight],
            ['1.0', COLORS.primaryDark]
        ],
        hoverongaps: false
    };

    Plotly.newPlot('chart-correlation', [trace], {
        ...layoutTemplate,
        margin: { t: 20, r: 20, b: 120, l: 120 },
        xaxis: { tickangle: 45 }
    }, { responsive: true, displayModeBar: false });
}

function renderTemporal(view) {
    const data = appData.temporal[view];
    let x, y;
    
    if (view === 'season') {
        const labels = { "0": "Winter", "1": "Spring", "2": "Summer", "3": "Fall" };
        x = Object.keys(data).map(k => labels[k] || k);
        y = Object.values(data);
    } else {
        x = Object.keys(data);
        y = Object.values(data);
    }

    const trace = {
        x: x,
        y: y,
        type: 'bar',
        marker: { color: COLORS.primary }
    };

    Plotly.newPlot('chart-temporal', [trace], {
        ...layoutTemplate,
        margin: { t: 10, r: 10, b: 40, l: 50 }
    }, { responsive: true, displayModeBar: false });
}

function renderHourly() {
    const data = appData.hourly;
    const hours = Object.keys(data).sort((a,b) => parseInt(a)-parseInt(b));
    
    const traces = hours.map(h => {
        const stats = data[h];
        // Plotly requires raw data for box plot unless we manually supply q1, median, etc via specific format
        // Since plotly 2.0 we can provide precomputed box stats using 'q1', 'median', 'q3', 'lowerfence', 'upperfence' arrays
        return {
            type: 'box',
            name: h,
            q1: [stats.q1],
            median: [stats.median],
            q3: [stats.q3],
            lowerfence: [stats.min],
            upperfence: [stats.max],
            marker: { color: COLORS.primary },
            line: { width: 1 }
        };
    });

    Plotly.newPlot('chart-hourly', traces, {
        ...layoutTemplate,
        showlegend: false,
        margin: { t: 10, r: 10, b: 40, l: 50 },
        xaxis: { title: 'Hour of Day', tickmode: 'linear' },
        yaxis: { title: 'Arrival Delay (min)' }
    }, { responsive: true, displayModeBar: false });
}

function setupListeners() {
    // Delay Dist
    document.getElementById('btn-dist-orig').addEventListener('click', (e) => {
        document.getElementById('btn-dist-yj').classList.remove('active');
        e.target.classList.add('active');
        renderDelayDist('original');
    });
    document.getElementById('btn-dist-yj').addEventListener('click', (e) => {
        document.getElementById('btn-dist-orig').classList.remove('active');
        e.target.classList.add('active');
        renderDelayDist('yeo_johnson');
    });

    // Categorical
    document.getElementById('btn-cat-pre').addEventListener('click', (e) => {
        document.getElementById('btn-cat-post').classList.remove('active');
        e.target.classList.add('active');
        renderCategorical('pre');
    });
    document.getElementById('btn-cat-post').addEventListener('click', (e) => {
        document.getElementById('btn-cat-pre').classList.remove('active');
        e.target.classList.add('active');
        renderCategorical('post');
    });

    // Temporal
    document.getElementById('select-temporal').addEventListener('change', (e) => {
        renderTemporal(e.target.value);
    });
}

document.addEventListener('DOMContentLoaded', initDashboard);
