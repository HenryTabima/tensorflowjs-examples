const tf = require('@tensorflow/tfjs')

const kgToLbs = kg => kg * 2.2

const xs = tf.tensor(Array.from({ length: 2000 }, (_, i) => i))
const ys = tf.tensor(Array.from({ length: 2000 }, (_, i) => kgToLbs(i)))

async function trainModel () {
  const model = tf.sequential()

  model.add(tf.layers.dense({
    units: 1,
    inputShape: 1
  }))

  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'adam'
  })

  await model.fit(xs, ys, {
    epochs: 500,
    shuffle: true
  })

  return model
}

trainModel().then(model => {
  console.log('Model trained')
  const lbs = model
    .predict(tf.tensor([10]))
    .asScalar()
    .dataSync()
  console.log(`10 Kg to Lbs: ${lbs}`)
})
