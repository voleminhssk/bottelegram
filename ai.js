const brain=require("brain.js")
const fs=require("fs")

const net=new brain.NeuralNetwork()

net.fromJSON(
    JSON.parse(
        fs.readFileSync("model.json")
    )
)

function predict(history){

    const last10=history.slice(-10)

    const input=last10.map(
        v=>v=="Tài"?1:0
    )

    const out=net.run(input)[0]

    const result=
        out>0.5?"Tài":"Xỉu"

    const confidence=
        out>0.5?
        out*100:
        (1-out)*100

    return{
        result,
        confidence:confidence.toFixed(2)
    }

}

module.exports=predict
