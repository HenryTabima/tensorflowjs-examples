import * as tf from '@tensorflow/tfjs'

const kgToLbs = kg => kg * 2.2

const xs = tf.tensor(Array.from({ length: 2000 }, (_, i) => i))
const ys = tf.tensor(Array.from({ length: 2000 }, (_, i) => kgToLbs(i)))

async function trainModel () {
  const model = tf.sequential()

  model.add(tf.layer.dense({
    units: 1,
    inputShape: 1
  }))

  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'adam'
  })

  await model.fit(xs, ys, {
    epochs: 100,
    shuffle: true
  })
}

trainModel().then(() => {
  console.log('Model trained')
})
