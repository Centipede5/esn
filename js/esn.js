var ESN = function(){
    var obj = this;
    obj.internal_weights = null;
    obj.n_internal_units = 0;
    obj.input_weights = null;
    obj.n_inputs = 0;
    obj.output_weights = null;
    obj.n_outputs = 0;
    obj.node_states = null;
    obj.leak_rate = 0.5;
    obj.ridge = new Ridge(0.1);
    obj.init = function(input_weights,internal_weights,output_weights){
        obj.n_inputs = input_weights[0].length;
        obj.n_internal_units = internal_weights.length;
        obj.n_outputs = output_weights.length;
        obj.input_weights = input_weights;
        obj.output_weights = output_weights;
        obj.internal_weights = internal_weights;
        obj.node_states = new Array(obj.n_internal_units).fill(0);
    }
    obj.update = function(input){
        // copy node states with the name "new node states"
        var new_node_states = new Array(obj.n_internal_units).fill(0);
        // update internal node states
        for(var i=0;i<obj.n_internal_units;i++){
            var sum = obj.node_states[i];
            // add inputs
            for(var j=0;j<obj.n_inputs;j++){
                sum += obj.input_weights[i][j]*input[j];
            }
            // add internal weights
            for(var j=0;j<obj.n_internal_units;j++){
                sum += obj.internal_weights[i][j]*obj.node_states[j];
            }
            // leaky integrator
            sum = obj.leak_rate*sum; //+ (1-obj.leak_rate)*obj.node_states[i];
            // squash with tanh
            new_node_states[i] = Math.tanh(sum);
        }
        // update node states
        obj.node_states = new_node_states;
        return obj.node_states;
    }
    // this function runs after update
    obj.predict = function(input){
        return obj.ridge.predict([obj.node_states])[0];
    }
    obj.clearNodeStates = function(){
        for(var i=0;i<obj.n_internal_units;i++){
            obj.node_states[i] = 0;
        }
    }
    obj.train = function(inputs,outputs,ndrop){
        var X = [];
        var y = [];
        for(var i=0;i<inputs.length;i++){
            obj.update(inputs[i]);
            if(i >= ndrop){
                X.push(obj.node_states);
                y.push(outputs[i]);
            }
        }
        obj.ridge.train(X,y);
        console.log('training score: ',obj.ridge.score(X,y));
    }
    obj.run = function(inputs){
        var y = [];
        for(var i=0;i<inputs.length;i++){
            obj.update(inputs[i]);
            y.push(obj.predict());
        }
        return y;
    }
    // convert to dot diagram
    obj.toDot = function(){
        // input nodes (filled yellow)
        var dot = "digraph ESN {\n";
        var radius = 2;
        for(var i=0;i<obj.n_inputs;i++){
            x = 0;
            y = radius*2*i/obj.n_inputs-radius;
            dot += "  in"+i+" [style=filled,fillcolor=yellow,pos=\""+x+","+y+"!\"];\n";
        }
        // internal nodes (filled blue) (make these circular using the pos attribute)
        
        for(var i=0;i<obj.n_internal_units;i++){
            var angle = i*2*Math.PI/obj.n_internal_units;
            var x = Math.round(radius*Math.cos(angle))+radius*2;
            var y = Math.round(radius*Math.sin(angle));
            var colorHex;
            if(window.colorInterpolation){
                colorHex = window.colorInterpolation((obj.node_states[i]+1)/2)
            }else{
                var intensity = Math.round(255*(obj.node_states[i]+1)/2).toString(16);
                if(intensity.length == 1) intensity = "0"+intensity;
                colorHex = "#"+intensity+intensity+intensity; // blue to red
            }
            dot += "  int"+i+" [style=filled,fillcolor=\""+colorHex+"\",pos=\""+x+","+y+"!\"];\n";
        }
        // connections between input and internal nodes (red)
        // for(var i=0;i<obj.n_inputs;i++){
        //     for(var j=0;j<obj.n_internal_units;j++){
        //         if(obj.input_weights[j][i] != 0){
        //             dot += "  in"+i+" -> int"+j+" [color=red];\n";
        //         }
        //     }
        // }
        // connections between internal nodes (black) 
        for(var i=0;i<obj.n_internal_units;i++){
            for(var j=0;j<obj.n_internal_units;j++){
                if(obj.internal_weights[j][i] != 0){
                    var color = obj.internal_weights[j][i] > 0 ? "green" : "red";
                    dot += "  int"+i+" -> int"+j+" [color="+color+",penwidth="+2*Math.abs(obj.internal_weights[j][i])+"];\n";
                }
            }
        }

        dot += "}";
        return dot;
    }
    return obj;
}
window.ESN = ESN;