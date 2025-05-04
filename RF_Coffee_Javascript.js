// 1. Defines the Region of interest
var roi = ee.Geometry.Polygon([
    [37.050749, -0.145285],
    [37.054869, -0.400714],
    [37.212797, -0.395221],
    [37.229277, -0.183737],
    [37.149626, -0.117819]
]).buffer(1000); // 1km buffer

// 2. The section loads multi-year Sentinel-2 Data
var startYear = 2020;
var endYear = 2023;
var years = ee.List.sequence(startYear, endYear);

function createAnnualComposite(year) {
  year = ee.Number(year);
  var startDate = ee.Date.fromYMD(year, 1, 1);
  var endDate = ee.Date.fromYMD(year, 12, 31);
  
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(startDate, endDate)
    .filterBounds(roi)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10));
  
  var withIndices = collection.map(function(image) {
    var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
    var ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI');
    var evi = image.expression(
      '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
        'NIR': image.select('B8'),
        'RED': image.select('B4'),
        'BLUE': image.select('B2')
    }).rename('EVI');
    
    return image
      .addBands(ndvi)
      .addBands(ndmi)
      .addBands(evi)
      .select(['B2','B3','B4','B8','B11','NDVI','NDMI','EVI'])
      .set('system:time_start', image.date().millis());
  });
  
  return withIndices.median()
    .set('year', year)
    .set('system:time_start', startDate.millis());
}

var annualComposites = ee.ImageCollection.fromImages(
  years.map(createAnnualComposite)
);

// 3. Definition of land over the classes
var trainingPolygons = ee.FeatureCollection([
  ee.Feature(ee.Geometry.Polygon([[37.10,-0.20],[37.11,-0.20],[37.11,-0.21],[37.10,-0.21]]), {'class': 1, 'label': 'Forest'}),
  ee.Feature(ee.Geometry.Polygon([[37.12,-0.18],[37.13,-0.18],[37.13,-0.19],[37.12,-0.19]]), {'class': 2, 'label': 'Deforested'}),
  ee.Feature(ee.Geometry.Polygon([[37.15,-0.20],[37.16,-0.20],[37.16,-0.21],[37.15,-0.21]]), {'class': 3, 'label': 'Agriculture'}),
  ee.Feature(ee.Geometry.Polygon([[37.18,-0.25],[37.19,-0.25],[37.19,-0.26],[37.18,-0.26]]), {'class': 4, 'label': 'Urban'})
]);

// 4. Training classifier for the model
var recentComposite = annualComposites.filter(ee.Filter.eq('year', endYear)).first();
var bands = ['B2','B3','B4','B8','B11','NDVI','NDMI','EVI'];

var training = recentComposite.select(bands)
  .sampleRegions({
    collection: trainingPolygons,
    properties: ['class'],
    scale: 10,
    tileScale: 8
  });

// Accuracy assessment for the prediction model for Confusion Matrix, Producer's Accuracy per class, Kappa Coefficient and the Overall accuracy
var withRandom = training.randomColumn('random');
var trainingSet = withRandom.filter(ee.Filter.gte('random', 0.3));
var validationSet = withRandom.filter(ee.Filter.lt('random', 0.3));

var classifier = ee.Classifier.smileRandomForest(100).train({
  features: trainingSet,
  classProperty: 'class',
  inputProperties: bands
});

var validated = validationSet.classify(classifier);
var confusionMatrix = validated.errorMatrix('class', 'classification');

print('Confusion Matrix (Validation):', confusionMatrix);
print('Overall Accuracy (Validation):', confusionMatrix.accuracy());
print('Kappa Coefficient:', confusionMatrix.kappa());
print('Producers Accuracy per Class:', confusionMatrix.producersAccuracy());

// 5. Classification for each year
var classifiedCollection = annualComposites.map(function(image) {
  return image.select(bands)
    .classify(classifier)
    .set('year', image.get('year'))
    .set('system:time_start', image.get('system:time_start'));
});

// 6. Calculation of the overall deforestation
var classifiedList = classifiedCollection.toList(classifiedCollection.size());
var forestStart = ee.Image(classifiedList.get(0)).eq(1);
var forestEnd = ee.Image(classifiedList.get(classifiedList.size().subtract(1))).eq(1);
var deforestation = forestStart.and(forestEnd.not()).rename('deforestation');

// 7. NDVI Trend calculation
var ndviWithTime = annualComposites.map(function(img) {
  var year = ee.Number(img.get('year'));
  var timeBand = ee.Image.constant(year).toFloat().rename('year');
  return img.select('NDVI').addBands(timeBand);
});

var trend = ndviWithTime.select(['year', 'NDVI']).reduce(ee.Reducer.linearFit());
var slope = trend.select('scale').rename('slope');

// 8. Visualization details
Map.centerObject(roi, 12);
Map.addLayer(deforestation, {palette: ['000000', 'FF0000']}, 'Deforestation 2020-2023');
Map.addLayer(slope, {
  min: -0.05, 
  max: 0.05,
  palette: ['FF0000', 'FFFFFF', '00FF00']
}, 'NDVI Trend (slope)');

var classVis = {min:1, max:4, palette:['darkgreen','brown','yellow','gray']};
for (var i=0; i<4; i++) {
  Map.addLayer(ee.Image(classifiedList.get(i)), classVis, 'Classification '+(startYear+i));
}

// 9. EXPORTS
Export.image.toDrive({
  image: deforestation,
  description: 'Deforestation_Map',
  folder: 'GEE_Exports',
  fileNamePrefix: 'deforestation_2020-2023',
  region: roi,
  scale: 10,
  maxPixels: 1e9
});

// 10. NDVI Time Series graph
var ndviChart = ui.Chart.image.seriesByRegion({
  imageCollection: annualComposites.select('NDVI'),
  regions: roi,
  reducer: ee.Reducer.mean(),
  scale: 100,
  xProperty: 'system:time_start'
}).setChartType('LineChart').setOptions({
  title: 'NDVI Time Series (with Labels)',
  vAxis: {title: 'NDVI'},
  hAxis: {title: 'Year', format: '####'},
  pointSize: 5,
  dataOpacity: 0.8,
  lineWidth: 2,
  annotations: {
    alwaysOutside: true,
    textStyle: {
      fontSize: 12,
      bold: true,
      color: '#000'
    }
  },
  series: {
    0: {
      pointShape: 'circle',
      color: '#1a9641',
      annotations: {style: 'line'}
    }
  },
  legend: {position: 'none'}
});
print(ndviChart);

// Deforestation area
var areaHa = deforestation.multiply(ee.Image.pixelArea())
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: roi,
    scale: 10,
    maxPixels: 1e9
  }).get('deforestation');
print('Deforested area (ha):', ee.Number(areaHa).divide(10000));

// 11. Legend for the output map
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px'
  }
});

legend.add(ui.Label({
  value: 'Map Legend',
  style: {fontWeight: 'bold', fontSize: '16px', margin: '0 0 6px 0'}
}));

function makeRow(color, name) {
  var colorBox = ui.Label({
    style: {
      backgroundColor: color,
      padding: '8px',
      margin: '0 8px 0 0'
    }
  });

  var description = ui.Label({
    value: name,
    style: {margin: '0', fontSize: '14px'}
  });

  return ui.Panel({
    widgets: [colorBox, description],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
}

// Classification colors
legend.add(ui.Label({value: 'Land Cover (Classification)', style: {fontWeight: 'bold', margin: '8px 0 4px 0'}}));
legend.add(makeRow('darkgreen', 'Forest'));
legend.add(makeRow('brown', 'Deforested'));
legend.add(makeRow('yellow', 'Agriculture / Coffee'));
legend.add(makeRow('gray', 'Urban'));

// Deforestation
legend.add(ui.Label({value: 'Deforestation 2020-2023', style: {fontWeight: 'bold', margin: '8px 0 4px 0'}}));
legend.add(makeRow('FF0000', 'Deforested Area'));
legend.add(makeRow('000000', 'No Change'));

// NDVI Trend
legend.add(ui.Label({value: 'NDVI Trend', style: {fontWeight: 'bold', margin: '8px 0 4px 0'}}));
legend.add(makeRow('00FF00', 'Vegetation Gain'));
legend.add(makeRow('FFFFFF', 'No Change'));
legend.add(makeRow('FF0000', 'Vegetation Loss'));

Map.add(legend);